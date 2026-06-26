import torch
from PIL import Image
import os, json, glob, random
from transformers import AutoProcessor, Idefics3ForConditionalGeneration
from peft import PeftModel

# ================================================================
# 1. SETUP & PATHS
# ================================================================
MODEL_ID = "HuggingFaceTB/SmolVLM-Base"
ADAPTER_PATH = "./smolvlm_turtlebot_vla_temporal"  # Where your model was saved
DATA_ROOT = "vla_prompt_training"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading Base Model: {MODEL_ID}")
processor = AutoProcessor.from_pretrained(MODEL_ID)

# Load base model in bfloat16 (Standard for A100/3090/Orin)
model = Idefics3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    #_attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
    trust_remote_code=True
).to(DEVICE)

# Load the LoRA adapter we just trained
print(f"Loading LoRA Adapter from: {ADAPTER_PATH}")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

# ================================================================
# 2. SELECT TEST DATA
# ================================================================
# Grab a random image from your training folders to see if it learned the scene
json_files = glob.glob(os.path.join(DATA_ROOT, "*/*.json"))
test_json = random.choice(json_files)

with open(test_json, "r") as f:
    meta = json.load(f)

img_path = os.path.join(os.path.dirname(test_json), meta["image"])
image = Image.open(img_path).convert("RGB")

# ================================================================
# 3. RUN THE "MOMENTUM" TEST
# ================================================================
# We will test the SAME image with two DIFFERENT histories to see if context matters.
test_scenarios = [
    {
        "name": "Steady Forward Momentum",
        "history": "forward -> forward -> forward -> forward -> forward",
        "instruction": meta.get("instruction", "Navigate to the door")
    },
    {
        "name": "Mid-Turn Context",
        "history": "forward -> turn_left -> turn_left -> turn_left -> turn_left",
        "instruction": meta.get("instruction", "Navigate to the door")
    }
]

print(f"\n--- SANITY CHECK START ---")
print(f"Testing Image: {img_path}")
print(f"Ground Truth Action: {meta.get('command_translated')}\n")

for scenario in test_scenarios:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"Task: {scenario['instruction']}. History: {scenario['history']}"}
            ]
        }
    ]

    # Apply template and generate
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(DEVICE)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False, # We want the most likely prediction
            temperature=0.0 
        )

    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    
    print(f"Scenario: {scenario['name']}")
    print(f"Input History: {scenario['history']}")
    print(f"Model Prediction: {generated_text.split('Assistant:')[-1].strip()}")
    print("-" * 30)

print("\nCheck if the model's 'Next Trajectory' differs based on the history!")