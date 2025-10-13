"""
Fine-tune SmolVLM on TurtleBot4 frames
Goal: Image → robot_action text (no question)
Compatible with macOS (CPU or Apple Silicon)
"""

import os, json, glob, re, torch
from PIL import Image
from datasets import Dataset
from transformers import (
    AutoProcessor, Idefics3ForConditionalGeneration,
    TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model

# ================================================================
# 1. PREPARE DATASET (index-pair JPG[i] ↔ JSON[i])
# ================================================================
print("Loading metadata...")

data_dir = os.path.join(os.getcwd(), "captured_frames/captured_frames")

# Natural sort key so "2.jpg" < "10.jpg"
def natural_key(path):
    base = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", base)]

# Collect images (jpg/jpeg/png) and JSONs
img_files = []
for pat in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
    img_files.extend(glob.glob(os.path.join(data_dir, pat)))
img_files = sorted(set(img_files), key=natural_key)

json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")), key=natural_key)

if not img_files or not json_files:
    raise RuntimeError(f"Found {len(img_files)} images and {len(json_files)} JSONs in {data_dir}")

# Pair strictly by index; truncate to shorter length if counts differ
min_len = min(len(img_files), len(json_files))
if len(img_files) != len(json_files):
    print(f"Warning: {len(img_files)} images vs {len(json_files)} JSONs. Truncating to {min_len} pairs.")

img_files = img_files[:min_len]
json_files = json_files[:min_len]

# Optional: quick preview of first few pairings
for i in range(min(3, min_len)):
    print(f"Pair[{i}]: {os.path.basename(img_files[i])} <-> {os.path.basename(json_files[i])}")

examples = []
for img_path, meta_path in zip(img_files, json_files):
    try:
        with open(meta_path, "r") as f:
            m = json.load(f)
        caption = m.get("robot_action")
        if not isinstance(caption, str) or not caption.strip():
            print(f"Skipping (empty/missing 'robot_action'): {os.path.basename(meta_path)}")
            continue
        if not os.path.exists(img_path):
            print(f"Skipping (missing image): {img_path}")
            continue
        examples.append({"image": img_path, "caption": caption.strip()})
    except Exception as e:
        print(f"Skipping pair due to JSON error ({os.path.basename(meta_path)}): {e}")
        continue

if len(examples) == 0:
    raise RuntimeError("No valid samples after pairing. Check JSON contents and keys.")

dataset = Dataset.from_list(examples)
train_ds, eval_ds = dataset, dataset

print(f"Loaded {len(train_ds)} training and {len(eval_ds)} validation samples.\n")

# ================================================================
# 2. MODEL + PROCESSOR SETUP
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

# Simple LoRA configuration (no bitsandbytes)
lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules=['down_proj','o_proj','k_proj','q_proj','gate_proj','up_proj','v_proj'],
    init_lora_weights="gaussian"
)
model = get_peft_model(model, lora_config)
print("Trainable parameters:", model.get_nb_trainable_parameters(), "\n")

# ================================================================
# 3. COLLATE FUNCTION
# ================================================================
image_token_id = processor.tokenizer.additional_special_tokens_ids[
    processor.tokenizer.additional_special_tokens.index("<image>")
]

def collate_fn(examples):
    texts, images, labels = [], [], []

    for ex in examples:
        image = Image.open(ex["image"]).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Frame captured by TurtleBot. Respond with only the exact robot_action value."},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": ex["caption"]},
                ],
            },
        ]

        # Create the conversation text
        conversation = processor.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False
        )
        texts.append(conversation)
        images.append(image)

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    # Copy labels from input_ids
    labels = batch["input_ids"].clone()

    # Mask <pad> tokens
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Mask <image> tokens
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    labels[labels == image_token_id] = -100

    # 🧩 Here's the critical fix:
    # Mask out everything *before* the assistant turn (i.e., the user input)
    for i, text in enumerate(texts):
        user_text = text.split("Assistant:")[0]
        user_token_count = len(processor.tokenizer(user_text).input_ids)
        labels[i, :user_token_count] = -100

    batch["labels"] = labels
    return batch




# Sanity test the collator
print("Testing collate_fn with one sample...")
try:
    _ = collate_fn([train_ds[0]])
    print("Collate function working.\n")
except Exception as e:
    raise RuntimeError(f"Collate function failed: {e}")

# ================================================================
# 4. TRAINING SETUP
# ================================================================
print("Setting up trainer...")

training_args = TrainingArguments(
    num_train_epochs=10,               # 🔥 more epochs — memorize the tiny dataset
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,     # no accumulation — update each step
    learning_rate=5e-4,                # higher LR = faster memorization
    warmup_steps=0,
    weight_decay=0.0,                  # no regularization
    logging_steps=1,                   # log every step
    save_strategy="no",                # skip checkpoint overhead
    fp16=torch.cuda.is_available(),
    output_dir="./smolvlm_turtlebot_action_overfit",
    report_to="none",
    remove_unused_columns=False,
    gradient_checkpointing=False,
    max_steps=-1
)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_ds,
    eval_dataset=eval_ds
)

print("Trainer summary:")
print("Train samples:", len(trainer.train_dataset))
print("Eval samples:", len(trainer.eval_dataset))
print("Using collate function:", trainer.data_collator.__name__, "\n")

# ================================================================
# 5. TRAIN
# ================================================================
print("Starting training...")
trainer.train()

# ================================================================
# 6. SAVE MODEL LOCALLY
# ================================================================
print("Saving model...")
trainer.save_model("./smolvlm_turtlebot_action_ft")
processor.save_pretrained("./smolvlm_turtlebot_action_ft")
print("Fine-tuning complete.\n")

# ================================================================
# 7. QUICK TEST
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
