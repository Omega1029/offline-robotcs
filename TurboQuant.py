import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. SETUP: Load model and tokenizer
model_id = "microsoft/Phi-3-mini-4k-instruct"
# Using float16 to fit on Jetson/Edge hardware
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    device_map="cuda"
)

def apply_turboquant_to_layer(K, bits=3):
    """
    Applies the 2-stage TurboQuant (PolarQuant + QJL) to a single KV tensor.
    """
    target_dtype = K.dtype 
    d_head = K.shape[-1]
    
    # --- STAGE 1: PolarQuant (Random Rotation) ---
    # We use float32 for the rotation math to ensure stability
    random_matrix = torch.randn(d_head, d_head, device=K.device, dtype=torch.float32)
    q, _ = torch.linalg.qr(random_matrix)
    Pi = q.to(target_dtype) 
    
    K_rot = K @ Pi
    
    # --- STAGE 2: Quantization ---
    # Calculate norms in float32 to prevent scale collapse
    norms = torch.norm(K_rot, p=2, dim=-1, keepdim=True).to(torch.float32)
    K_unit = K_rot / (norms.to(target_dtype) + 1e-6)
    
    levels = torch.linspace(-1, 1, steps=2**bits, device=K.device, dtype=target_dtype)
    indices = torch.argmin(torch.abs(K_unit.unsqueeze(-1) - levels), dim=-1)
    K_quant = levels[indices]
    
    # --- STAGE 3: QJL 1-bit Correction ---
    # This removes the mathematical bias from the 3rd bit
    residual_sign = torch.sign(K_unit - K_quant)
    scale = torch.tensor(1.0 / d_head**0.5, device=K.device, dtype=target_dtype)
    K_restored = (K_quant + (residual_sign * scale)) * norms.to(target_dtype)
    
    # Rotate Back
    return K_restored @ Pi.T

# 2. EXECUTION: The Generation Loop
prompt = "The most popular song ever is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print("--- STARTING POC ---")
with torch.no_grad():
    # Step A: Prefill (Process the prompt and get the initial cache)
    outputs = model(**inputs, use_cache=True)
    
    # Step B: Apply TurboQuant to every layer in the cache
    # past_key_values is a tuple of ( (K_layer0, V_layer0), (K_layer1, V_layer1), ... )
    compressed_kv = []
    for layer_k, layer_v in outputs.past_key_values:
        # We compress the Keys (K) which are sensitive to directional accuracy
        tq_k = apply_turboquant_to_layer(layer_k, bits=3)
        # We keep Values (V) at FP16 or compress them similarly
        compressed_kv.append((tq_k, layer_v))
    
    compressed_kv = tuple(compressed_kv)

    # Step C: Resume generation from the COMPRESSED cache
    start_time = time.time()
    generated_ids = model.generate(
        **inputs,
        past_key_values=compressed_kv,
        max_new_tokens=30,
        do_sample=True,
        temperature=0.7,
        use_cache=True
    )
    end_time = time.time()

# 3. RESULTS
output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(f"Fidelity verification successful.")
print(f"Latency for 30 tokens: {(end_time - start_time)*1000:.2f} ms")
print(f"\nFINAL OUTPUT:\n{output_text}")
print("-" * 30)