import cv2
import torch
import os
from peft import PeftModel
from transformers import AutoProcessor, Idefics3ForConditionalGeneration
from PIL import Image
import numpy as np
from time import time

# ================================================================
#  CONFIG
# ================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "./smolvlm_turtlebot_action_ft"
BASE_MODEL_ID = "HuggingFaceTB/SmolVLM-Base"
CAMERA_INDEX = 0  # Change if you have multiple cameras

# ================================================================
#  LOAD MODEL + PROCESSOR
# ================================================================
print(f"Loading fine-tuned model from: {MODEL_DIR}")

processor = AutoProcessor.from_pretrained(MODEL_DIR)

base_model = Idefics3ForConditionalGeneration.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16 if DEVICE != "cpu" else torch.float32,
).to(DEVICE)

model = PeftModel.from_pretrained(base_model, MODEL_DIR)
model = model.merge_and_unload()  # Merge LoRA for inference
model.eval()
print("✅ Model and LoRA adapter loaded.\n")

# ================================================================
#  HELPER: RUN INFERENCE ON FRAME
# ================================================================
def predict_action(frame):
    """Run SmolVLA inference on a single OpenCV frame and return predicted text."""
    # Convert to RGB PIL image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Frame captured by TurtleBot. Respond with only the exact robot_action value, e.g., forward_0.2_3.0s."}
            ],
        },
    ]

    # Build prompt + encode
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(images=[img], text=[prompt], return_tensors="pt").to(DEVICE)

    # Generate
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=15)

    result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    # Clean up: only return the assistant output if “Assistant:” exists
    if "Assistant:" in result:
        result = result.split("Assistant:")[-1].strip()
    return result

# ================================================================
#  MAIN LOOP (OPENCV)
# ================================================================
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError("❌ Could not open camera")

print("🎥 Press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame capture failed — exiting.")
        break

    start = time()
    predicted_action = predict_action(frame)
    command = predicted_action.split('_')
    action = command[0]
    distance = command[1]
    time_limit = command[2]
    print(f"Action: {predicted_action}, Command: {action}, {distance}, {time_limit}")
    end = time()

    # Overlay prediction text
    fps = 1 / (end - start)
    cv2.putText(frame, f"Action: {predicted_action}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    # Show frame
    cv2.imshow("SmolVLA Live Inference", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Stream closed.")
