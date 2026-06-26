import os, json, glob, re, torch, random, math
from collections import defaultdict
from PIL import Image
from datasets import Dataset
from transformers import (
    AutoProcessor, Idefics3ForConditionalGeneration,
    TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model

# Check for torchvision for augmentations
try:
    import torchvision.transforms as T
    from torchvision.transforms import functional as TF
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

# -----------------------
# Hyperparams
# -----------------------
UPSAMPLE_FACTOR = 1
EPOCHS = 1 * UPSAMPLE_FACTOR
SEED = 42

MODEL_ID = "HuggingFaceTB/SmolVLM-Base"
OUTPUT_DIR = "./smolvlm_turtlebot_vla"
DATA_ROOT = "vla_prompt_training"

random.seed(SEED)
torch.manual_seed(SEED)

# ================================================================
# 1. DATA LOADING (Nested Directory Support)
# ================================================================
print(f"Scanning {DATA_ROOT} for data...")

dataset_list = []
# Match all JSON files in any subfolder of vla_prompt_training
json_paths = glob.glob(os.path.join(DATA_ROOT, "**/*.json"), recursive=True)

for jp in json_paths:
    try:
        with open(jp, "r") as f:
            meta = json.load(f)
        
        # In your schema: 'image' is just the filename, likely in same folder as JSON
        img_filename = meta.get("image")
        img_path = os.path.join(os.path.dirname(jp), img_filename)
        
        instruction = meta.get("instruction")
        target_action = meta.get("command_translated") # e.g. "left_0.8_0.2s"

        if os.path.exists(img_path) and target_action:
            dataset_list.append({
                "image": img_path,
                "instruction": instruction,
                "caption": target_action,
                # Extract action type for balancing (left, right, forward, etc.)
                "action_type": target_action.split("_")[0].lower()
            })
    except Exception as e:
        print(f"Error parsing {jp}: {e}")

# ================================================================
# 2. BALANCING & SPLITTING
# ================================================================
actions_dict = defaultdict(list)
for item in dataset_list:
    actions_dict[item["action_type"]].append(item)

print("\nDistribution before balancing:")
for k, v in actions_dict.items():
    print(f"  {k}: {len(v)} samples")

# Updated counts from your last run
counts = {
    "forward": 18468,
    "left": 2344,
    "right": 1262,
    "backward": 149
}
############################################################################
##
##
##        WEIGHTED LOSS SECTION
##
##
#############################################################################
class WeightedActionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 1. Log-dampened weights: [forward, left, right, backward]
        # Math: 1 + ln(normalized_weight)
        # Result: [1.0, 3.0, 3.7, 5.8] 
        # This is much safer for LoRA stability.
        weights = torch.tensor([1.0, 3.0, 3.7, 5.8])#, device=model.device)
        
        action_ids = inputs.pop("action_ids")
        outputs = model(**inputs)
        
        if self.args.n_gpu > 1:
            loss = outputs.get("loss")
        else:
            batch_weights = weights[action_ids]
            # Scale the mean loss by the dampened importance of this batch
            loss = outputs.get("loss") * batch_weights.mean()

        return (loss, outputs) if return_outputs else loss

# Updated counts from your last run
counts = {
    "forward": 18468,
    "left": 2344,
    "right": 1262,
    "backward": 149
}

# Calculate inverse frequency normalized to the majority class
total = sum(counts.values())
raw_weights = {k: (total / v) for k, v in counts.items()}
norm_factor = raw_weights["forward"]
# Final weights: {'forward': 1.0, 'left': 7.88, 'right': 14.63, 'backward': 123.95}
action_weights = {k: math.sqrt(v / norm_factor) for k, v in raw_weights.items()}




#################################################################3
#
#
#
#                      ENDS
#
#
###################################################################


# 1. Flatten all categories into a single list (No downsampling)
all_samples = []
for action_type, samples in actions_dict.items():
    all_samples.extend(samples)

# 2. Shuffle the entire dataset
random.shuffle(all_samples)

# 3. Perform the split (85/15)
split_idx = int(0.85 * len(all_samples))
train_examples = all_samples[:split_idx]
eval_examples = all_samples[split_idx:]

# 4. Create Datasets
train_ds = Dataset.from_list(train_examples)
eval_ds = Dataset.from_list(eval_examples)

print(f"Total samples found: {len(all_samples)}")
print(f"Final training set size: {len(train_ds)}")
print(f"Final evaluation set size: {len(eval_ds)}")

# ================================================================
# 3. MODEL SETUP
# ================================================================
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Idefics3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if DEVICE != "cpu" else torch.float32,
    trust_remote_code=True
).to(DEVICE)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=['down_proj','o_proj','k_proj','q_proj','gate_proj','up_proj','v_proj'],
    init_lora_weights="gaussian"
)
model = get_peft_model(model, lora_config)

# ================================================================
# 4. AUGMENTATION & COLLATOR
# ================================================================
def maybe_flip_and_swap(image: Image.Image, caption: str):
    """Flip image horizontally and swap 'left'/'right' in caption string."""
    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if "left" in caption and "right" not in caption:
            caption = caption.replace("left", "right")
        elif "right" in caption and "left" not in caption:
            caption = caption.replace("right", "left")
    return image, caption

# Add this mapping right above collate_fn
ACTION_TO_ID = {"forward": 0, "left": 1, "right": 2, "backward": 3}

def collate_fn(examples):
    texts, images = [], []
    # We'll store the prompt lengths to know where the assistant starts
    prompt_lengths = []

    for ex in examples:
        image = Image.open(ex["image"]).convert("RGB")
        caption = ex["caption"]
        image, caption = maybe_flip_and_swap(image, caption)

        # 1. Build the User Prompt part (the part we want to MASK)
        user_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"Instruction: {ex['instruction']}"}
                ]
            }
        ]
        
        # 2. Build the Full Conversation (the part we want to TRAIN on)
        full_messages = user_messages + [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": f"Action: {caption}"}]
            }
        ]

        # Use apply_chat_template for both
        user_prompt = processor.apply_chat_template(user_messages, add_generation_prompt=True, tokenize=False)
        full_prompt = processor.apply_chat_template(full_messages, add_generation_prompt=False, tokenize=False)

        texts.append(full_prompt)
        images.append(image)

        # We need to know how many tokens are in the user-only part
        # We use the processor's tokenizer directly on the prompt string
        user_tokenized = processor.tokenizer(user_prompt, add_special_tokens=False).input_ids
        prompt_lengths.append(len(user_tokenized))

    # 3. Process the batch
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    
    labels = batch["input_ids"].clone()
    
    # 4. Mask the labels
    for i in range(len(texts)):
        # Mask everything from the beginning up to the end of the user prompt
        # This keeps ONLY the assistant's "Action: ..." part unmasked
        labels[i, :prompt_lengths[i]] = -100
        
        # Also mask padding tokens at the end of the sequence
        # (Where attention_mask is 0)
        labels[i, batch["attention_mask"][i] == 0] = -100

    batch["labels"] = labels
    #print(f"Unmasked labels in batch: {(batch['labels'] != -100).sum().item()}")
    batch["action_ids"] = torch.tensor([ACTION_TO_ID[ex["action_type"]] for ex in examples])
    return batch

# ================================================================
# 5. TRAIN
# ================================================================
training_args = TrainingArguments(
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=5,
    output_dir=OUTPUT_DIR,
    save_strategy="no",
    fp16=(DEVICE == "cuda"),
    remove_unused_columns=False,
    push_to_hub=False
)

#trainer = Trainer(

trainer = WeightedActionTrainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_ds,
    eval_dataset=eval_ds
)

print("\nStarting Fine-tuning...")
trainer.train()

# Save
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"✅ Model saved to {OUTPUT_DIR}")