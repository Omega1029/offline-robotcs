import os, json, glob, torch, random, math, base64, time
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

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

# -----------------------
# Hyperparams
# -----------------------
UPSAMPLE_FACTOR = 1
EPOCHS = 1 * UPSAMPLE_FACTOR
SEED = 42

MODEL_ID = "HuggingFaceTB/SmolVLM-Base"
OUTPUT_DIR = "./smolvlm_turtlebot_vla"
DATA_ROOT = "vla_prompt_training"

# Scene descriptions: "smolvlm" (local captioning before LoRA), "openai" (vision API), or "cache" (JSON + cache file only)
DESCRIPTION_BACKEND = os.environ.get("VLA_DESC_BACKEND", "smolvlm")
OPENAI_VISION_MODEL = os.environ.get("OPENAI_VISION_MODEL", "gpt-4o-mini")
DESCRIPTION_CACHE_PATH = os.path.join(DATA_ROOT, "image_descriptions_cache.json")
SCENE_CAPTION_PROMPT = (
    "Describe the scene in 1–2 short sentences for a wheeled robot: obstacles, open paths, and notable objects."
)
DESC_BATCH_SIZE = int(os.environ.get("VLA_DESC_BATCH_SIZE", "4"))

random.seed(SEED)
torch.manual_seed(SEED)


def _load_desc_cache(path):
    if os.path.isfile(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save_desc_cache(path, mapping):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(mapping, f, indent=2)
    os.replace(tmp, path)


def describe_scene_openai(image_path, client, model_name, prompt):
    with open(image_path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode("ascii")
    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png" if ext == ".png" else "image/jpeg"
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }
        ],
        max_tokens=200,
    )
    return (resp.choices[0].message.content or "").strip()


def describe_paths_smolvlm_batch(model, processor, paths, device, prompt, batch_size):
    """Returns list of strings aligned with paths."""
    out = []
    for i in range(0, len(paths), batch_size):
        chunk = paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in chunk]
        messages_list = [
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            for _ in chunk
        ]
        texts = [
            processor.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            for m in messages_list
        ]
        inputs = processor(text=texts, images=images, return_tensors="pt").to(device)
        in_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )
        new_tokens = generated_ids[:, in_len:]
        for row in range(new_tokens.shape[0]):
            text = processor.tokenizer.decode(new_tokens[row], skip_special_tokens=True).strip()
            out.append(text)
    return out


def ensure_scene_descriptions(dataset_list, backend, processor, model, device, cache_path, prompt):
    """
    Sets item['scene_description'] for each row using JSON fields, cache file, or generation.
    backend: 'smolvlm' | 'openai' | 'cache'
    """
    cache = _load_desc_cache(cache_path)
    need_paths = []
    for item in dataset_list:
        d = (item.get("scene_description") or "").strip()
        if d:
            continue
        p = item["image"]
        if p in cache and cache[p].strip():
            item["scene_description"] = cache[p].strip()
            continue
        need_paths.append(p)

    unique_missing = list(dict.fromkeys(need_paths))
    if not unique_missing:
        _save_desc_cache(cache_path, cache)
        return

    if backend == "cache":
        for p in unique_missing:
            cache[p] = cache.get(p, "")
        for item in dataset_list:
            if not (item.get("scene_description") or "").strip():
                item["scene_description"] = cache.get(item["image"], "")
        _save_desc_cache(cache_path, cache)
        print(f"WARNING: {len(unique_missing)} images have no description (backend=cache). Training with empty Scene lines.")
        return

    if backend == "openai":
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai (`pip install openai`) for DESCRIPTION_BACKEND=openai")
        client = OpenAI()
        for p in tqdm(unique_missing, desc="OpenAI scene descriptions"):
            try:
                cache[p] = describe_scene_openai(p, client, OPENAI_VISION_MODEL, prompt)
                time.sleep(0.05)
            except Exception as e:
                print(f"OpenAI describe failed for {p}: {e}")
                cache[p] = ""
    elif backend == "smolvlm":
        model.eval()
        for i in tqdm(range(0, len(unique_missing), DESC_BATCH_SIZE), desc="SmolVLM scene descriptions"):
            chunk = unique_missing[i : i + DESC_BATCH_SIZE]
            texts = describe_paths_smolvlm_batch(model, processor, chunk, device, prompt, len(chunk))
            for path, t in zip(chunk, texts):
                cache[path] = t
            _save_desc_cache(cache_path, cache)
        model.train()
    else:
        raise ValueError(f"Unknown DESCRIPTION_BACKEND: {backend}")

    for item in dataset_list:
        if not (item.get("scene_description") or "").strip():
            item["scene_description"] = cache.get(item["image"], "").strip()

    _save_desc_cache(cache_path, cache)
    print(f"Scene descriptions ready ({len(cache)} paths in cache).")

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
        scene_desc = (meta.get("scene_description") or meta.get("image_description") or "").strip()

        if os.path.exists(img_path) and target_action:
            dataset_list.append({
                "json_path": jp,
                "image": img_path,
                "instruction": instruction,
                "caption": target_action,
                "scene_description": scene_desc,
                # Extract action type for balancing (left, right, forward, etc.)
                "action_type": target_action.split("_")[0].lower()
            })
    except Exception as e:
        print(f"Error parsing {jp}: {e}")

# ================================================================
# 1b. SCENE DESCRIPTIONS (JSON / cache / SmolVLM / OpenAI vision)
# ================================================================
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = None

if DESCRIPTION_BACKEND == "smolvlm":
    model = Idefics3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if DEVICE != "cpu" else torch.float32,
        trust_remote_code=True,
    ).to(DEVICE)
    ensure_scene_descriptions(
        dataset_list,
        "smolvlm",
        processor,
        model,
        DEVICE,
        DESCRIPTION_CACHE_PATH,
        SCENE_CAPTION_PROMPT,
    )
elif DESCRIPTION_BACKEND == "openai":
    ensure_scene_descriptions(
        dataset_list,
        "openai",
        processor,
        None,
        DEVICE,
        DESCRIPTION_CACHE_PATH,
        SCENE_CAPTION_PROMPT,
    )
elif DESCRIPTION_BACKEND == "cache":
    ensure_scene_descriptions(
        dataset_list,
        "cache",
        processor,
        None,
        DEVICE,
        DESCRIPTION_CACHE_PATH,
        SCENE_CAPTION_PROMPT,
    )
else:
    raise ValueError(
        f"DESCRIPTION_BACKEND must be smolvlm, openai, or cache; got {DESCRIPTION_BACKEND!r}"
    )

if model is None:
    model = Idefics3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if DEVICE != "cpu" else torch.float32,
        trust_remote_code=True,
    ).to(DEVICE)

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
# 3. LoRA
# ================================================================
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
def maybe_flip_and_swap(image: Image.Image, caption: str, scene_desc: str = ""):
    """Flip image horizontally and swap 'left'/'right' in caption and scene text."""
    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if "left" in caption and "right" not in caption:
            caption = caption.replace("left", "right")
        elif "right" in caption and "left" not in caption:
            caption = caption.replace("right", "left")
        if scene_desc:
            if "left" in scene_desc and "right" not in scene_desc:
                scene_desc = scene_desc.replace("left", "right")
            elif "right" in scene_desc and "left" not in scene_desc:
                scene_desc = scene_desc.replace("right", "left")
    return image, caption, scene_desc

# Add this mapping right above collate_fn
ACTION_TO_ID = {"forward": 0, "left": 1, "right": 2, "backward": 3}

def collate_fn(examples):
    texts, images = [], []
    # We'll store the prompt lengths to know where the assistant starts
    prompt_lengths = []

    for ex in examples:
        image = Image.open(ex["image"]).convert("RGB")
        caption = ex["caption"]
        scene_desc = (ex.get("scene_description") or "").strip()
        image, caption, scene_desc = maybe_flip_and_swap(image, caption, scene_desc)
        user_text = f"Instruction: {ex['instruction']}"
        if scene_desc:
            user_text += f"\nScene: {scene_desc}"

        # 1. Build the User Prompt part (the part we want to MASK)
        user_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text}
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