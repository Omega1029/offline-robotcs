from llama_cpp import Llama
from PIL import Image
import time
import argparse
import os
import glob
import random
import numpy as np

# ==========================================================
# LLM LOAD
# ==========================================================

llm = Llama(
    model_path="smolvlm-satellite-captioning-Q4_K_M.gguf",
    mmproj_path="mmproj-smolvlm-satellite-captioning-f16.gguf",
    n_gpu_layers=99,
    n_ctx=512,
    verbose=False,
)

# ==========================================================
# RANDOM IMAGE SAMPLER
# ==========================================================

def get_random_image(base_dir="vla_prompt_training", balanced=False):
    task_dirs = ["task_dock", "task_door", "task_printer"]

    if balanced:
        task = random.choice(task_dirs)
        pattern = os.path.join(base_dir, task, "*.jpg")
        images = glob.glob(pattern)

        if not images:
            raise ValueError(f"No images found in {task}")

        return random.choice(images)

    all_images = []
    for task in task_dirs:
        pattern = os.path.join(base_dir, task, "*.jpg")
        all_images.extend(glob.glob(pattern))

    if not all_images:
        raise ValueError("No images found in dataset!")

    return random.choice(all_images)

# ==========================================================
# IMAGE RESIZE (SmolVLM Native 384x384)
# ==========================================================

def resize_to_384(input_path, output_path="temp_384.jpg"):
    img = Image.open(input_path)
    resized_img = img.resize((384, 384), Image.Resampling.LANCZOS)
    resized_img.save(output_path)
    return output_path

# ==========================================================
# INFERENCE FUNCTION
# ==========================================================

def infer_robot_action(image_path: str, prompt: str) -> str:
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"file://{os.path.abspath(image_path)}"
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
        max_tokens=12,
        temperature=0.0,
        stop=["<end_of_utterance>"],
    )

    return response["choices"][0]["message"]["content"].strip()

# ==========================================================
# ARGUMENTS
# ==========================================================

ap = argparse.ArgumentParser()
ap.add_argument(
    "--prompt", "-p",
    default="Predict robot_action. ONE action only."
)
ap.add_argument(
    "--runs", "-r",
    type=int,
    default=20
)
ap.add_argument(
    "--balanced",
    action="store_true"
)

args = ap.parse_args()

# ==========================================================
# BENCHMARK
# ==========================================================

print("\nStarting LiteVLA Benchmark (Auto-Resize Enabled)\n")

times = []

# Warm-up
for _ in range(3):
    img_path = get_random_image(balanced=args.balanced)
    resized = resize_to_384(img_path)
    _ = infer_robot_action(resized, args.prompt)

# Benchmark
for i in range(args.runs):
    img_path = get_random_image(balanced=args.balanced)
    resized_path = resize_to_384(img_path)

    start = time.perf_counter()
    ans = infer_robot_action(resized_path, args.prompt)
    end = time.perf_counter()

    latency = end - start
    times.append(latency)

    print(f"[{i+1}/{args.runs}] {os.path.basename(img_path)} → {ans} | {latency:.4f}s")

print("\n==============================")
print("Mean latency:", np.mean(times))
print("Std latency :", np.std(times))
print("Min latency :", np.min(times))
print("Max latency :", np.max(times))
print("==============================\n")

# Optional: Clean up temp file
if os.path.exists("temp_384.jpg"):
    os.remove("temp_384.jpg")
