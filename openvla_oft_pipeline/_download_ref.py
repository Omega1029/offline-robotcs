from huggingface_hub import snapshot_download
import time
t=time.time()
p = snapshot_download(
    "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10",
    local_dir="checkpoints/openvla_oft_reference/sogd10",
    ignore_patterns=["lora_adapter/*", ".gitattributes"],  # merged weights already in model.safetensors
    max_workers=8,
)
print(f"DONE in {(time.time()-t)/60:.1f} min -> {p}", flush=True)
