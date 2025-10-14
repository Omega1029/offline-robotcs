from peft import PeftModel, PeftConfig
from transformers import AutoProcessor, Idefics3ForConditionalGeneration
import torch
from PIL import Image
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================================================================
#  Load processor and base model
# ================================================================
ft_dir = "./smolvlm_turtlebot_action_ft"
print(f"Loading fine-tuned model from: {ft_dir}")

# Load processor
processor = AutoProcessor.from_pretrained(ft_dir)

# Load base model
base_model_id = "HuggingFaceTB/SmolVLM-Base"
model = Idefics3ForConditionalGeneration.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16 if DEVICE != "cpu" else torch.float32,
).to(DEVICE)

# Attach LoRA adapter weights
model = PeftModel.from_pretrained(model, ft_dir)
model = model.merge_and_unload()  # optional: merge LoRA for inference only
model.eval()
print("✅ Model and LoRA adapter loaded.\n")

print("Running quick inference test...")

test_img = "captured_frames/captured_frames/frame_000016_20250604_144135_525.jpg"  # or train_ds[0]["image"]
if not os.path.exists(test_img):
    raise FileNotFoundError
    
img = Image.open(test_img).convert("RGB")

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
