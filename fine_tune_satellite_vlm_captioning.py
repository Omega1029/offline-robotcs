import ast
import csv
import io
import json
import os
import random
import tokenize
import torch
from PIL import Image

from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoProcessor,
    Idefics3ForConditionalGeneration,
    TrainingArguments,
    Trainer,
)

try:
    import torchvision.transforms as T

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
OUTPUT_DIR = "./smolvlm_satellite_captioning"
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "satellite"))

FIXED_INSTRUCTION = "Describe this satellite image in detail."
MAX_SEQUENCE_LENGTH = 2048

random.seed(SEED)
torch.manual_seed(SEED)


def parse_caption_strings(raw: str) -> list[str]:
    raw = raw.strip()
    if not raw.startswith("["):
        return []
    out: list[str] = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(raw).readline):
            if tok.type == tokenize.STRING:
                out.append(ast.literal_eval(tok.string))
    except (tokenize.TokenError, SyntaxError, ValueError):
        return []
    return out


def load_split(csv_path: str, data_root: str):
    """Load rows from train.csv / valid.csv / test.csv into training records."""
    records = []
    parse_fail = 0
    missing_img = 0
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fp = row["filepath"].strip()
            captions = parse_caption_strings(row["captions"])
            if not captions:
                parse_fail += 1
                continue
            img_path = os.path.join(data_root, fp)
            if not os.path.isfile(img_path):
                missing_img += 1
                continue
            records.append(
                {
                    "image": os.path.abspath(img_path),
                    "captions": captions,
                    "instruction": FIXED_INSTRUCTION,
                }
            )
    return records, parse_fail, missing_img


def split_basenames(records):
    return {os.path.basename(r["image"]) for r in records}


def maybe_color_jitter(image: Image.Image) -> Image.Image:
    if not HAS_TORCHVISION or random.random() >= 0.5:
        return image
    jitter = T.ColorJitter(
        brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05
    )
    return jitter(image)


# ================================================================
# 1. DATA LOADING (CSV + satellite image paths)
# ================================================================
train_csv = os.path.join(DATA_ROOT, "train.csv")
valid_csv = os.path.join(DATA_ROOT, "valid.csv")
test_csv = os.path.join(DATA_ROOT, "test.csv")

print(f"Loading data from {DATA_ROOT} ...")
train_records, train_parse_fail, train_missing = load_split(train_csv, DATA_ROOT)
valid_records, valid_parse_fail, valid_missing = load_split(valid_csv, DATA_ROOT)
# test.csv held out for downstream eval; load only if you want overlap stats
_test_records, test_parse_fail, test_missing = load_split(test_csv, DATA_ROOT)

print(
    f"train: {len(train_records)} rows (parse_fail={train_parse_fail}, missing_image={train_missing})"
)
print(
    f"valid: {len(valid_records)} rows (parse_fail={valid_parse_fail}, missing_image={valid_missing})"
)
print(
    f"test (held out): {len(_test_records)} rows (parse_fail={test_parse_fail}, missing_image={test_missing})"
)

bt, bv, btest = (
    split_basenames(train_records),
    split_basenames(valid_records),
    split_basenames(_test_records),
)
overlap_tv = bt & bv
overlap_vt = bv & btest
overlap_tt = bt & btest
if overlap_tv or overlap_vt or overlap_tt:
    print(
        f"Filename overlap (basenames): train∩valid={len(overlap_tv)}, valid∩test={len(overlap_vt)}, train∩test={len(overlap_tt)}"
    )

if not train_records:
    raise SystemExit(
        f"No training samples: place images under {DATA_ROOT} (paths from train.csv) or fix CSVs."
    )
if not valid_records:
    raise SystemExit(
        f"No validation samples: place images under {DATA_ROOT} (paths from valid.csv) or fix CSVs."
    )

train_ds = Dataset.from_list(train_records)
eval_ds = Dataset.from_list(valid_records)

# ================================================================
# 2. MODEL SETUP
# ================================================================
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Idefics3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if DEVICE != "cpu" else torch.float32,
    trust_remote_code=True,
).to(DEVICE)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=[
        "down_proj",
        "o_proj",
        "k_proj",
        "q_proj",
        "gate_proj",
        "up_proj",
        "v_proj",
    ],
    init_lora_weights="gaussian",
)
model = get_peft_model(model, lora_config)

# ================================================================
# 3. COLLATOR (mask user tokens; random caption per step from multi-ref list)
# ================================================================
def collate_fn(examples):
    texts, images = [], []
    prompt_lengths = []

    proc_kw = dict(truncation=True, max_length=MAX_SEQUENCE_LENGTH)

    for ex in examples:
        image = Image.open(ex["image"]).convert("RGB")
        caption = random.choice(ex["captions"])
        image = maybe_color_jitter(image)

        user_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"Instruction: {ex['instruction']}"},
                ],
            }
        ]
        full_messages = user_messages + [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": f"Caption: {caption}"}],
            }
        ]

        user_prompt = processor.apply_chat_template(
            user_messages, add_generation_prompt=True, tokenize=False
        )
        full_prompt = processor.apply_chat_template(
            full_messages, add_generation_prompt=False, tokenize=False
        )

        texts.append(full_prompt)
        images.append(image)

        # Length must use the same image-token expansion as the full batch (not raw tokenizer).
        user_enc = processor(
            text=[user_prompt],
            images=[image],
            return_tensors="pt",
            padding=False,
            **proc_kw,
        )
        prompt_lengths.append(int(user_enc["input_ids"].shape[1]))

    batch = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        **proc_kw,
    )

    labels = batch["input_ids"].clone()
    for i in range(len(texts)):
        pl = min(prompt_lengths[i], labels.shape[1])
        labels[i, :pl] = -100
        labels[i, batch["attention_mask"][i] == 0] = -100

    batch["labels"] = labels
    return batch


# ================================================================
# 4. TRAIN
# ================================================================
training_args = TrainingArguments(
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=5,
    output_dir=OUTPUT_DIR,
    save_strategy="no",
    eval_strategy="epoch",
    fp16=(DEVICE == "cuda"),
    remove_unused_columns=False,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
)

print("\nStarting fine-tuning...")
trainer.train()

model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

cfg_path = os.path.join(OUTPUT_DIR, "satellite_training_config.json")
with open(cfg_path, "w") as f:
    json.dump(
        {
            "model_id": MODEL_ID,
            "data_root": DATA_ROOT,
            "fixed_instruction": FIXED_INSTRUCTION,
            "seed": SEED,
            "epochs": EPOCHS,
            "caption_strategy": "random_choice_per_collate_from_multi_reference",
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "train_rows": len(train_records),
            "valid_rows": len(valid_records),
        },
        f,
        indent=2,
    )

print(f"Model and processor saved to {OUTPUT_DIR}")
print(f"Training config written to {cfg_path}")
