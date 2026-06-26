import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch.nn.functional as F
import csv
from pathlib import Path

# 1. Load using the correct AutoClass
model_id = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

def turboquant_vlm_logic(tensor, bits=3):
    dt, dev = tensor.dtype, tensor.device
    b, h, s, d = tensor.shape

    # Stage 1: Random Rotation
    R = torch.randn(d, d, device=dev, dtype=torch.float32)
    Q, _ = torch.linalg.qr(R)
    Pi = Q.to(dt)
    
    t_rot = tensor.view(-1, d) @ Pi

    # Stage 2: Quantization (with float32 norms for long-seq stability)
    norms = torch.norm(t_rot, p=2, dim=-1, keepdim=True).to(torch.float32)
    t_unit = t_rot / (norms.to(dt) + 1e-6)
    
    levels = torch.linspace(-1, 1, steps=2**bits, device=dev, dtype=dt)
    idx = torch.argmin(torch.abs(t_unit.unsqueeze(-1) - levels), dim=-1)
    t_quant = levels[idx]

    # Stage 3: QJL Correction (The secret to VLM focal accuracy)
    res_sign = torch.sign(t_unit - t_quant)
    scale = 1.0 / (d**0.5)
    t_restored = (t_quant + (res_sign * scale)) * norms.to(dt)
    
    return (t_restored @ Pi.T).view(b, h, s, d)

# 2. Real satellite inference test
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

with torch.no_grad():
    res = model(**inputs, use_cache=True)
    # SmolVLM cache: past_key_values[layer][0 for Keys]
    original_k = res.past_key_values[0][0] 
    
    compressed_k = turboquant_vlm_logic(original_k, bits=3)

    fidelity = F.cosine_similarity(original_k.flatten().float(), compressed_k.flatten().float(), dim=0)
    print(f"--- SmolVLM TurboQuant Verification ---")
    print(f"Fidelity: {fidelity.item():.6f}")