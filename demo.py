from llama_cpp import Llama
import argparse
import numpy as np
import time

ap = argparse.ArgumentParser()
ap.add_argument("--prompt", "-p", help="Enter your prompt", required=False, default="Predict robot_action. ONE action only.")
ap.add_argument("--limit", "-l", help="Limit for rune", required=False, type=int, default=10)
ap.add_argument(
    "--image",
    "-i",
    help="Path to input image",
    default="test_robot_384.jpg",
)
ap.add_argument(
    "--model",
    "-m",
    help="Path to GGUF model file",
    default="litevla-finetuned-Q4_K_M.gguf",
)
ap.add_argument(
    "--mmproj",
    "-mm",
    help="Path to multimodal projection (mmproj) GGUF",
    default="mmproj-litevla-f16.gguf",
)
args = ap.parse_args()

llm = Llama(
    model_path=args.model,
    mmproj_path=args.mmproj,
    n_gpu_layers=99,
    n_ctx=2048,        # You don't need 2048 for robotics
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
        max_tokens=32,
        temperature=0.0,
        stop=["<end_of_utterance>"],
    )

    return response["choices"][0]["message"]["content"].strip()


times = []
limit = args.limit
ans = infer_robot_action(image_path=args.image, prompt=args.prompt)
# Warm-up
for _ in range(3):
    ans = infer_robot_action(image_path=args.image, prompt=args.prompt)

for _ in range(limit):
    start = time.perf_counter()
    ans = infer_robot_action(image_path=args.image, prompt=args.prompt)
    end = time.perf_counter()
    times.append(end - start)

print("Mean latency:", np.mean(times))
print("Std latency:", np.std(times))
print("Caption Predicted", ans)
