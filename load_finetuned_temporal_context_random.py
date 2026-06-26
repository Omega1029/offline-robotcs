import os, json, glob, re, torch, random
from PIL import Image
from transformers import AutoProcessor, Idefics3ForConditionalGeneration
from peft import PeftModel

# ================================================================
# 1. CONFIG & MODEL LOADING
# ================================================================
MODEL_ID = "HuggingFaceTB/SmolVLM-Base"
ADAPTER_PATH = "./smolvlm_turtlebot_vla_temporal"
DATA_ROOT = "vla_prompt_training"
HISTORY_SIZE = 10
CHUNK_SIZE = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading Model and Adapter on {DEVICE}...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
base_model = Idefics3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    #_attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
    trust_remote_code=True
).to(DEVICE)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# Helper for natural sorting (so 10.json comes after 2.json)
def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

# ================================================================
# 2. DATASET NAVIGATOR
# ================================================================
def get_random_test_case():
    # 1. Pick a random task subdirectory
    subdirs = [d for d in glob.glob(os.path.join(DATA_ROOT, "*")) if os.path.isdir(d)]
    selected_dir = random.choice(subdirs)
    
    # 2. Get all JSONs in that folder, sorted chronologically
    json_paths = sorted(glob.glob(os.path.join(selected_dir, "*.json")), key=natural_key)
    
    # 3. Pick a random index (ensure we aren't at the very end so we have a Ground Truth chunk)
    # We want at least CHUNK_SIZE frames left to compare
    max_idx = max(0, len(json_paths) - CHUNK_SIZE - 1)
    target_idx = random.randint(0, max_idx)
    
    # 4. Reconstruct History
    history_actions = []
    for h_idx in range(target_idx - HISTORY_SIZE, target_idx):
        if h_idx < 0:
            history_actions.append("none")
        else:
            with open(json_paths[h_idx], "r") as f:
                history_actions.append(json.load(f).get("command_raw", "none"))
    
    # 5. Get Target Metadata and Image
    with open(json_paths[target_idx], "r") as f:
        meta = json.load(f)
    
    # 6. Get Ground Truth Chunk (The actual next 10 actions from the data)
    gt_actions = []
    for g_idx in range(target_idx, target_idx + CHUNK_SIZE):
        with open(json_paths[g_idx], "r") as f:
            gt_actions.append(json.load(f).get("command_translated"))

    return {
        "image_path": os.path.join(selected_dir, meta["image"]),
        "instruction": meta.get("instruction", "Navigate"),
        "history": " -> ".join(history_actions),
        "ground_truth": " | ".join(gt_actions)
    }

# ================================================================
# 3. RUN INFERENCE
# ================================================================
test_case = get_random_test_case()
image = Image.open(test_case["image_path"]).convert("RGB")

prompt_text = f"Task: {test_case['instruction']}. History: {test_case['history']}"

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt_text}
        ]
    }
]

prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(DEVICE)

print("\n" + "="*50)
print(f"TESTING IMAGE: {os.path.basename(test_case['image_path'])}")
print(f"PROMPT SENT: {prompt_text}")
print("="*50)

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False
    )

prediction = processor.batch_decode(outputs, skip_special_tokens=True)[0].split("Assistant:")[-1].strip()

print(f"\n[GROUND TRUTH CHUNK]:\n{test_case['ground_truth']}")
print(f"\n[MODEL PREDICTION]:\n{prediction}")

# SIMPLE METRIC: How many actions match exactly?
gt_list = test_case['ground_truth'].split(" | ")
pred_list = prediction.replace("Next Trajectory: ", "").split(" | ")

matches = 0
for g, p in zip(gt_list, pred_list):
    if g.strip() == p.strip(): matches += 1

print(f"\nAccuracy for this chunk: {matches}/{len(gt_list)} actions matched perfectly.")