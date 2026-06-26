from llama_cpp import Llama
import time
import argparse

llm = Llama(
    model_path="smolvlm-finetuned-Q4_K_M.gguf",
    mmproj_path="mmproj-smolvlm-f16.gguf",
    n_gpu_layers=99,
    n_ctx=512,        # You don't need 2048 for robotics
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


ap = argparse.ArgumentParser()
ap.add_argument("--prompt","-p", help="Enter your prompt", required=False, default="Predict robot_action. ONE action only.")
args = ap.parse_args()

start = time.time()
print(infer_robot_action(image_path="test_robot_384.jpg", prompt=args.prompt))
end = time.time()
elapsed_time = round(end-start, 2)
print("It Took {}s".format(elapsed_time))
