"""
Fine-tune SmolVLM on TurtleBot4 frames (Balanced Action Dataset)
Goal: Equal representation of each action type (forward, left, right, stop, etc.)
"""

import os, json, glob, re, torch, random
from collections import defaultdict
from PIL import Image
from datasets import Dataset
from transformers import (
    AutoProcessor, Idefics3ForConditionalGeneration,
    TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model

EPOCHS = 1
#TARGET_COUNT = 2990  # Match the 'right' class size


# ================================================================
# 1. PREPARE DATASET (index-pair JPG[i] ↔ JSON[i])
# ================================================================
print("Loading metadata...")

data_dir = os.path.join(os.getcwd(), "captured_frames/captured_frames")

# Natural sort key so "2.jpg" < "10.jpg"
def natural_key(path):
    base = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", base)]

# Collect all image and json files
img_files = []
for pat in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
    img_files.extend(glob.glob(os.path.join(data_dir, pat)))
img_files = sorted(set(img_files), key=natural_key)
json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")), key=natural_key)

if not img_files or not json_files:
    raise RuntimeError(f"Found {len(img_files)} images and {len(json_files)} JSONs in {data_dir}")

min_len = min(len(img_files), len(json_files))
if len(img_files) != len(json_files):
    print(f"Warning: {len(img_files)} images vs {len(json_files)} JSONs. Truncating to {min_len} pairs.")

img_files, json_files = img_files[:min_len], json_files[:min_len]

# ================================================================
# 2. LOAD AND GROUP BY ACTION
# ================================================================
actions_dict = defaultdict(list)
for img_path, meta_path in zip(img_files, json_files):
    try:
        with open(meta_path, "r") as f:
            m = json.load(f)
        caption = m.get("robot_action")
        if not isinstance(caption, str) or not caption.strip():
            continue
        if not os.path.exists(img_path):
            continue

        # Extract action keyword (e.g., "forward" from "forward_0.2_3.0s")
        action_type = caption.split("_")[0].lower().strip()
        actions_dict[action_type].append({"image": img_path, "caption": caption.strip()})
    except Exception as e:
        print(f"Skipping pair due to JSON error ({os.path.basename(meta_path)}): {e}")
        continue

# Show distribution before balancing
print("\nAction distribution before balancing:")
for k, v in actions_dict.items():
    print(f"  {k}: {len(v)} samples")

# ================================================================
# 3. BALANCE CLASSES (REMOVE 'sequence', KEEP backward FULL, others → 2990)
# ================================================================


# Remove 'sequence' class entirely
random.seed(42)

# --- Balance by downsampling every class to the min count (excluding 'sequence')
if "sequence" in actions_dict:
    del actions_dict["sequence"]

class_counts = {k: len(v) for k, v in actions_dict.items()}
target = min(class_counts.values())  # simplest fair target

balanced = []
for k, v in actions_dict.items():
    v_shuf = v[:]  # copy
    random.shuffle(v_shuf)
    take = v_shuf[:target]
    for ex in take:
        ex["action_type"] = k
    balanced.extend(take)

# Stratified 85/15 split
train_examples, eval_examples = [], []
for k in actions_dict.keys():
    group = [e for e in balanced if e["action_type"] == k]
    random.shuffle(group)
    cut = int(0.85 * len(group))
    train_examples.extend(group[:cut])
    eval_examples.extend(group[cut:])

train_ds = Dataset.from_list(train_examples)
eval_ds  = Dataset.from_list(eval_examples)

print("Final per-class counts (train):",
      {k: sum(1 for e in train_examples if e["action_type"]==k) for k in actions_dict})
print("Final per-class counts (eval):",
      {k: sum(1 for e in eval_examples  if e["action_type"]==k) for k in actions_dict})
print("Total:", len(train_ds), "+", len(eval_ds))


# ================================================================
# 4. MODEL + PROCESSOR SETUP
# ================================================================
print("Loading model and processor...")

MODEL_ID = "HuggingFaceTB/SmolVLM-Base"
DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Using device: {DEVICE}")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Idefics3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if DEVICE != "cpu" else torch.float32
).to(DEVICE)

# LoRA setup
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=['down_proj','o_proj','k_proj','q_proj','gate_proj','up_proj','v_proj'],
    init_lora_weights="gaussian"
)
model = get_peft_model(model, lora_config)
print("Trainable parameters:", model.get_nb_trainable_parameters(), "\n")

# ================================================================
# 5. COLLATE FUNCTION
# ================================================================
def collate_fn(examples):
    texts, images = [], []
    for ex in examples:
        image = Image.open(ex["image"]).convert("RGB")
        messages = [
            {"role": "user", "content": [{"type": "image"},
                {"type":"text","text":"Frame captured by TurtleBot. Respond with only the exact robot_action value."}]},
            {"role": "assistant", "content": [{"type":"text","text": ex["caption"]}]},
        ]
        conversation = processor.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False
        )
        texts.append(conversation)
        images.append(image)

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    input_ids = batch["input_ids"]

    # Build labels: mask everything up to the start of the assistant’s reply
    labels = input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    labels[labels == image_token_id] = -100

    # Determine assistant start per sample by tokenizing the user-only portion
    for i, ex in enumerate(examples):
        # Rebuild the same messages but WITHOUT the assistant part to get its length
        user_only = [
            {"role": "user", "content": [{"type":"image"},
                {"type":"text","text":"Frame captured by TurtleBot. Respond with only the exact robot_action value."}]}
        ]
        user_text = processor.apply_chat_template(
            user_only, add_generation_prompt=False, tokenize=False
        )
        user_ids = processor.tokenizer(user_text, return_tensors="pt").input_ids.squeeze(0)
        user_len = user_ids.size(0)
        labels[i, :user_len] = -100

    batch["labels"] = labels
    return batch


_ = collate_fn([train_ds[0]])
print("✅ Collate function verified.\n")

# ================================================================
# 6. TRAINING SETUP
# ================================================================

training_args = TrainingArguments(
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-4,
    warmup_steps=0,
    weight_decay=0.0,
    logging_steps=1,
    save_strategy="no",
    fp16=torch.cuda.is_available(),
    output_dir="./smolvlm_turtlebot_action_balanced",
    report_to="none",
    remove_unused_columns=False,
    gradient_checkpointing=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_ds,
    eval_dataset=eval_ds
)

print("Trainer ready — starting training.\n")
trainer.train()

# ================================================================
# 7. SAVE MODEL
# ================================================================
print("\nSaving model...")
trainer.save_model("./smolvlm_turtlebot_action_balanced")
processor.save_pretrained("./smolvlm_turtlebot_action_balanced")
print("✅ Fine-tuning complete (balanced dataset).\n")

# ================================================================
# 8. QUICK TEST
# ================================================================
print("Running quick inference test...")

test_img = train_ds[0]["image"]
img = Image.open(test_img)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text",
             "text": "Frame captured by TurtleBot. Respond with only the exact robot_action value, e.g., forward_0.2_3.0s."}
        ],
    },
]

prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = processor(images=[img], text=[prompt], return_tensors="pt").to(DEVICE)

with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=15)

result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
print("Predicted:", result)
print("From image:", os.path.basename(test_img))
