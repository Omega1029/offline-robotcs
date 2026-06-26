import torch
from PIL import Image
from transformers import AutoProcessor, Idefics3ForConditionalGeneration
from peft import PeftModel

# -----------------------
# Config
# -----------------------
MODEL_ID = "HuggingFaceTB/SmolVLM-Base" # The base model
ADAPTER_DIR = "./smolvlm_turtlebot_vla" # Where your fine-tuned weights are
TEST_IMAGE_PATH = "vla_prompt_training/task_door/capture_1749217567306.jpg"
INSTRUCTION = "Find the printer"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================================================================
# 1. LOAD MODEL & PROCESSOR
# ================================================================
print(f"Loading base model: {MODEL_ID}...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

# Load the base model in FP16 (or FP32 if on CPU)
base_model = Idefics3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    trust_remote_code=True
).to(DEVICE)

print(f"Loading LoRA adapters from: {ADAPTER_DIR}...")
# This merges your 44-chunk fine-tuning into the base model
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR).to(DEVICE)
model.eval() # Set to evaluation mode

# ================================================================
# 2. PREPARE INPUT
# ================================================================
try:
    image = Image.open(TEST_IMAGE_PATH).convert("RGB")
except FileNotFoundError:
    print(f"Error: Could not find image at {TEST_IMAGE_PATH}. Please check the path.")
    exit()

# Format for Idefics3/SmolVLM chat template
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": f"Instruction: {INSTRUCTION}"}
        ]
    }
]

# Apply chat template and tokenize
text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = processor(text=[text_prompt], images=[image], return_tensors="pt").to(DEVICE)

# ================================================================
# 3. GENERATE
# ================================================================
print("\nGenerating Trajectory...")
with torch.no_grad():
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=16,
        do_sample=False, # Use greedy decoding for more consistent VLA outputs
        temperature=0.0
    )

# Decode output
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

print("-" * 30)
print(f"PROMPT: {INSTRUCTION}")
print(f"MODEL OUTPUT: {generated_text[0]}")
print("-" * 30)

# ================================================================
# 4. COHERENCE CHECK (PARSING TEST)
# ================================================================
output_str = generated_text[0]
if "Trajectory:" in output_str:
    trajectory = output_str.split("Trajectory:")[-1].strip()
    steps = trajectory.split(" | ")
    print(f"Detected {len(steps)} action steps.")
    for idx, step in enumerate(steps):
        print(f"  Step {idx+1}: {step}")
else:
    print("Warning: Model did not use the 'Trajectory:' prefix. Coherence may be low.")