import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


DEFAULT_MODEL_ID = "openvla/openvla-7b"
DEFAULT_LORA_DIR = Path("checkpoints/my_openvla_lora")
DEFAULT_UNNORM_KEY = "bridge_orig"


def find_sample_capture(root: Path = Path("captured_frames/captured_frames")):
    """Return a local image/instruction pair from the captured robot dataset."""
    for metadata_path in sorted(root.glob("*_metadata.json")):
        with metadata_path.open() as f:
            metadata = json.load(f)

        image_path = root / Path(metadata["filename"]).name
        if image_path.exists():
            action = metadata.get("robot_action", "move safely")
            instruction = f"perform the robot action {action}"
            return image_path, instruction

    for image_path in sorted(root.glob("*.jpg")):
        return image_path, "navigate safely while avoiding obstacles"

    raise FileNotFoundError(f"No sample images found under {root}")


def get_device(allow_cpu: bool = False) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")

    if allow_cpu:
        return torch.device("cpu")

    raise RuntimeError(
        "CUDA is not available. OpenVLA-7B normally needs a GPU notebook. "
        "Rerun with --allow-cpu only for a very slow smoke test."
    )


def load_openvla(
    model_id: str = DEFAULT_MODEL_ID,
    lora_dir: Path | None = DEFAULT_LORA_DIR,
    load_in_4bit: bool = False,
    allow_cpu: bool = False,
):
    device = get_device(allow_cpu=allow_cpu)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"Using device: {device}")
    print(f"Loading processor: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    model_kwargs = {
        "attn_implementation": "sdpa",
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }

    if load_in_4bit:
        if device.type != "cuda":
            raise RuntimeError("4-bit loading requires CUDA.")
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["device_map"] = "auto"

    print(f"Loading OpenVLA model: {model_id}")
    model = AutoModelForVision2Seq.from_pretrained(model_id, **model_kwargs)
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    if not load_in_4bit:
        model = model.to(device)

    if lora_dir and lora_dir.exists():
        from peft import PeftModel

        print(f"Applying LoRA adapter: {lora_dir}")
        model = PeftModel.from_pretrained(model, str(lora_dir))
    elif lora_dir:
        print(f"LoRA adapter not found, using base model only: {lora_dir}")

    model.eval()
    return processor, model, device, dtype


def predict_action(
    processor,
    model,
    image_path: str | Path,
    instruction: str,
    device: torch.device,
    dtype: torch.dtype,
    unnorm_key: str = DEFAULT_UNNORM_KEY,
):
    image = Image.open(image_path).convert("RGB")
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"
    inputs = processor(prompt, image).to(device, dtype=dtype)

    with torch.no_grad():
        action = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)

    return action


def parse_args():
    parser = argparse.ArgumentParser(description="Run an OpenVLA-7B inference smoke test.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--instruction")
    parser.add_argument("--lora-dir", type=Path, default=DEFAULT_LORA_DIR)
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--unnorm-key", default=DEFAULT_UNNORM_KEY)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    image_path = args.image
    instruction = args.instruction
    if image_path is None or instruction is None:
        sample_image, sample_instruction = find_sample_capture()
        image_path = image_path or sample_image
        instruction = instruction or sample_instruction

    processor, model, device, dtype = load_openvla(
        model_id=args.model_id,
        lora_dir=None if args.no_lora else args.lora_dir,
        load_in_4bit=args.load_in_4bit,
        allow_cpu=args.allow_cpu,
    )

    print(f"Image: {image_path}")
    print(f"Instruction: {instruction}")
    action = predict_action(
        processor,
        model,
        image_path=image_path,
        instruction=instruction,
        device=device,
        dtype=dtype,
        unnorm_key=args.unnorm_key,
    )
    print("Predicted action:")
    print(action)


if __name__ == "__main__":
    main()
