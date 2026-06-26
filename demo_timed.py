from llama_cpp import Llama
import time
import argparse
import numpy as np
import os
from collections import Counter
# -------------------------------
# Load model once (outside timing)
# -------------------------------
llm = Llama(
    model_path="litevla-finetuned_Q4_K_M.gguf",
    mmproj_path="mmproj-litevla-f16.gguf",
    n_gpu_layers=99,
    n_ctx=512,
    verbose=False,
)

def infer_robot_action(image_path: str, prompt: str) -> str:
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"file://{image_path}"
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

# -------------------------------
# CLI
# -------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--prompt", "-p",
                default="Predict robot_action. ONE action only. example forward_0.1s_0.3 or left_0.2s_0.4")
args = ap.parse_args()

image_path = "test_robot_384.jpg"

# Optional: preload image into OS cache
# Prevent first-read disk variance
with open(image_path, "rb") as f:
    _ = f.read()

# -------------------------------
# Warm-up (important)
# -------------------------------
for _ in range(5):
    infer_robot_action(image_path=image_path, prompt=args.prompt)

# -------------------------------
# Timed Runs
# -------------------------------
times = []

NUM_RUNS = 300
actions = []
for _ in range(NUM_RUNS):
    start = time.perf_counter()
    action = infer_robot_action(image_path=image_path, prompt=args.prompt)
    end = time.perf_counter()
    actions.append(action.split("_")[0])
    times.append(end - start)

times = np.array(times)

print(f"Runs: {NUM_RUNS}")
print(f"Mean latency: {times.mean():.6f} s")
print(f"Std latency:  {times.std():.6f} s")
print(f"Min latency:  {times.min():.6f} s")
print(f"Max latency:  {times.max():.6f} s")
print(f"Hz equivalent: {1.0 / times.mean():.2f} Hz")
print(f"Actions: {Counter(actions)}")