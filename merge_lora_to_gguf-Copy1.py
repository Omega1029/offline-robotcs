from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
import torch
import os

BASE_MODEL = "HuggingFaceTB/SmolVLM-Base"
LORA_DIR  = "smolvlm_turtlebot_action_ft"
OUT_DIR   = "smolvlm_merged_for_gguf"

assert os.path.exists(os.path.join(LORA_DIR, "adapter_config.json")), \
    "❌ adapter_config.json not found — wrong LoRA directory"

model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
    device_map="cpu"
)

model = PeftModel.from_pretrained(model, LORA_DIR)
model = model.merge_and_unload()

model.save_pretrained(OUT_DIR, safe_serialization=True)

processor = AutoProcessor.from_pretrained(BASE_MODEL)
processor.save_pretrained(OUT_DIR)

print("✅ LoRA merged into", OUT_DIR)

