from llama_cpp import Llama

llm = Llama(
    model_path="smolvlm-satellite-captioning-Q4_K_M.gguf",
    mmproj_path="mmproj-smolvlm-satellite-captioning-f16.gguf",
    n_gpu_layers=99,
    n_ctx=2048,
)

response = llm(
    prompt="Describe the image in detail.",
    image="baseballfield_85.jpg",
    max_tokens=6,
    temperature=0.0,
)

print(response["choices"][0]["text"])
