import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch.nn.functional as F
import csv
from pathlib import Path

# 1. SETUP: Load using the correct AutoClass
model_id = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

def turboquant_vlm_optimized(tensor, bits=3):
    """
    Applies VLM-optimized 2-stage TurboQuant with dynamic residual scaling
    to handle heavy-tailed vision token outliers.
    """
    dt, dev = tensor.dtype, tensor.device
    b, h, s, d = tensor.shape

    # --- Stage 1: PolarQuant (Random Rotation) ---
    R = torch.randn(d, d, device=dev, dtype=torch.float32)
    Q, _ = torch.linalg.qr(R)
    Pi = Q.to(dt)
    
    t_rot = tensor.view(-1, d) @ Pi

    # --- Stage 2: Quantization ---
    # Calculate norms in float32; use clamp to survive massive vision token spikes
    norms = torch.norm(t_rot, p=2, dim=-1, keepdim=True).to(torch.float32)
    t_unit = t_rot / torch.clamp(norms.to(dt), min=1e-6)
    
    levels = torch.linspace(-1, 1, steps=2**bits, device=dev, dtype=dt)
    idx = torch.argmin(torch.abs(t_unit.unsqueeze(-1) - levels), dim=-1)
    t_quant = levels[idx]

    # --- Stage 3: DYNAMIC QJL Correction ---
    # Dynamically measure the residual scale instead of using static 1/sqrt(d)
    residual = t_unit - t_quant
    dynamic_scale = residual.abs().mean(dim=-1, keepdim=True)
    
    res_sign = torch.sign(residual)
    t_restored = (t_quant + (res_sign * dynamic_scale)) * norms.to(dt)
    
    # Rotate Back
    return (t_restored @ Pi.T).view(b, h, s, d)


# 2. EXECUTION: Real satellite inference test
repo_root = Path(__file__).resolve().parent
csv_path = repo_root / "satellite" / "test.csv"

with csv_path.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    first_row = next(reader)
    rel_image_path = first_row["filepath"]

candidate_paths = [
    repo_root / "satellite" / rel_image_path,  # e.g. satellite/test/airport_348.jpg
    repo_root / rel_image_path,                # e.g. test/airport_348.jpg
]

image_path = next((p for p in candidate_paths if p.exists()), None)
if image_path is None:
    raise FileNotFoundError(
        f"Could not find satellite image '{rel_image_path}'. "
        "Expected under 'satellite/' or repo root. Extract/copy image files first."
    )

image = Image.open(image_path).convert("RGB")

# SmolVLM expects one <image> token in text for each provided image.
prompt = "<image>What is in view?"
inputs = processor(text=prompt, images=[image], return_tensors="pt").to("cuda")

print("--- STARTING POC ---")
with torch.no_grad():
    res = model(**inputs, use_cache=True)
    
    # SmolVLM cache structure: past_key_values[layer][0 for Keys, 1 for Values]
    original_k = res.past_key_values[0][0] 
    
    # Compress the keys
    compressed_k = turboquant_vlm_optimized(original_k, bits=3)

    # 3. RESULTS: Correct per-vector fidelity measurement
    # dim=-1 ensures we check similarity at the individual head_dim level
    cos_sim = F.cosine_similarity(original_k.float(), compressed_k.float(), dim=-1)
    fidelity = cos_sim.mean()
    
    print(f"--- SmolVLM TurboQuant Verification ---")
    print(f"Original K shape: {original_k.shape}")
    print(f"Compressed K shape: {compressed_k.shape}")
    print(f"Fidelity (Per-vector Mean): {fidelity.item():.6f}")
    print("-" * 39)