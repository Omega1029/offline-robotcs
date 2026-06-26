import torch, gc

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

cleanup()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Only let the script see one A100
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import os, json, glob, re, torch, random
from collections import defaultdict
from PIL import Image
from datasets import Dataset
from transformers import (
    AutoProcessor, Idefics3ForConditionalGeneration,
    TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model

# -----------------------
# Hyperparams
# -----------------------
UPSAMPLE_FACTOR = 1
EPOCHS = 5 * UPSAMPLE_FACTOR
SEED = 42
CHUNK_SIZE = 10     # Number of future actions to predict
HISTORY_SIZE = 5    # Number of past actions to provide as context

MODEL_ID = "HuggingFaceTB/SmolVLM-Base"
OUTPUT_DIR = "./smolvlm_turtlebot_vla_temporal"
DATA_ROOT = "vla_prompt_training"

random.seed(SEED)
torch.manual_seed(SEED)

# ================================================================
# 1. DATA LOADING (Temporal Context + Chunking)
# ================================================================
print(f"Scanning {DATA_ROOT} for sequences...")

dataset_list = []
subdirs = [d for d in glob.glob(os.path.join(DATA_ROOT, "*")) if os.path.isdir(d)]

for sdir in subdirs:
    # Ensure chronological order
    json_paths = sorted(glob.glob(os.path.join(sdir, "*.json")))
    
    for i, jp in enumerate(json_paths):
        try:
            # --- PART A: LOOK BACK (History) ---
            history_actions = []
            for h_idx in range(i - HISTORY_SIZE, i):
                if h_idx < 0:
                    history_actions.append("none") # Padding for start of task
                else:
                    with open(json_paths[h_idx], "r") as f:
                        meta = json.load(f)
                        history_actions.append(meta.get("command_raw", "none"))
            
            history_str = " -> ".join(history_actions)

            # --- PART B: LOOK AHEAD (Future Chunk) ---
            chunk_actions = []
            for j in range(i, i + CHUNK_SIZE):
                idx = min(j, len(json_paths) - 1)
                with open(json_paths[idx], "r") as f:
                    meta = json.load(f)
                    chunk_actions.append(meta.get("command_translated"))

            with open(jp, "r") as f:
                current_meta = json.load(f)
                img_path = os.path.join(sdir, current_meta.get("image"))
                instruction = current_meta.get("instruction", "Navigate the robot")

            if os.path.exists(img_path):
                dataset_list.append({
                    "image": img_path,
                    "instruction": instruction,
                    "history": history_str, # <--- NEW: State Context
                    "caption": " | ".join(chunk_actions),
                    "action_type": chunk_actions[0].split("_")[0].lower()
                })
        except Exception as e:
            print(f"Error parsing {jp}: {e}")

# ================================================================
# 2. BALANCING & SPLITTING
# ================================================================
actions_dict = defaultdict(list)
for item in dataset_list:
    actions_dict[item["action_type"]].append(item)

target_count = min(len(v) for v in actions_dict.values())
balanced_data = []
for k, v in actions_dict.items():
    random.shuffle(v)
    balanced_data.extend(v[:target_count])

random.shuffle(balanced_data)
split_idx = int(0.85 * len(balanced_data))
train_ds = Dataset.from_list(balanced_data[:split_idx])
eval_ds = Dataset.from_list(balanced_data[split_idx:])

# ================================================================
# 3. MODEL SETUP (PEFT/LoRA)
# ================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Idefics3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if DEVICE != "cuda" else torch.float32,
    #attn_implementation= "flash_attention_2" if DEVICE == "cuda" else "eager",
    trust_remote_code=True
).to(DEVICE)

lora_config = LoraConfig(
    r=16, 
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=['down_proj','o_proj','k_proj','q_proj','gate_proj','up_proj','v_proj'],
    init_lora_weights="gaussian"
)
model = get_peft_model(model, lora_config)

# ================================================================
# 4. COLLATOR (Modified for History Injection)
# ================================================================
def maybe_flip_and_swap(image: Image.Image, caption: str, history: str):
    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        # Flip directions in BOTH history and future caption
        def flip_dir(text):
            text = text.replace("left", "TEMP").replace("right", "left").replace("TEMP", "right")
            return text
        return image, flip_dir(caption), flip_dir(history)
    return image, caption, history

def collate_fn(examples):
    texts, images, prompt_lengths = [], [], []

    for ex in examples:
        image = Image.open(ex["image"]).convert("RGB")
        image, caption, history = maybe_flip_and_swap(image, ex["caption"], ex["history"])

        # Construct the prompt with History
        user_messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"Task: {ex['instruction']}. History: {history}"}
            ]
        }]
        
        full_messages = user_messages + [{
            "role": "assistant",
            "content": [{"type": "text", "text": f"Next Trajectory: {caption}"}]
        }]

        user_prompt = processor.apply_chat_template(user_messages, add_generation_prompt=True, tokenize=False)
        full_prompt = processor.apply_chat_template(full_messages, add_generation_prompt=False, tokenize=False)

        texts.append(full_prompt)
        images.append(image)
        
        # Masking logic
        user_tokenized = processor.tokenizer(user_prompt, add_special_tokens=False).input_ids
        prompt_lengths.append(len(user_tokenized))

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)#.to(DEVICE)
    labels = batch["input_ids"].clone()
    
    for i in range(len(texts)):
        labels[i, :prompt_lengths[i]] = -100 
        labels[i, batch["attention_mask"][i] == 0] = -100

    batch["labels"] = labels
    return batch

# ================================================================
# 5. TRAIN
# ================================================================
training_args = TrainingArguments(
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    output_dir=OUTPUT_DIR,
    save_strategy="no",
    fp16=(DEVICE == "cuda"),
    remove_unused_columns=False,
    logging_steps=10,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_ds,
    eval_dataset=eval_ds
)

print("Starting State-Aware VLA Training...")
trainer.train()
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"✅ State-Aware VLA Model saved to {OUTPUT_DIR}")