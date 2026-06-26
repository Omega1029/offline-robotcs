import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor


def load_model(args):
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is not available. Use --allow-cpu only for a very slow smoke test.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"Using device: {device}")
    print(f"Loading processor: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    model_kwargs = {
        "attn_implementation": args.attn_implementation,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }

    if args.load_in_4bit:
        if device.type != "cuda":
            raise RuntimeError("--load-in-4bit requires CUDA.")
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["device_map"] = "auto"

    print(f"Loading base model: {args.model_id}")
    model = AutoModelForVision2Seq.from_pretrained(args.model_id, **model_kwargs)
    if hasattr(model, "tie_weights"):
        model.tie_weights()

    if not args.load_in_4bit:
        model = model.to(device)

    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
        adapter_config = checkpoint_dir / "adapter_config.json"
        if adapter_config.exists():
            print(f"Applying LoRA adapter: {checkpoint_dir}")
            model = PeftModel.from_pretrained(model, str(checkpoint_dir))
            if args.merge_lora:
                print("Merging LoRA adapter into base model for inference")
                model = model.merge_and_unload()
        else:
            print(f"Loading merged checkpoint directly: {checkpoint_dir}")
            model = AutoModelForVision2Seq.from_pretrained(
                str(checkpoint_dir),
                attn_implementation=args.attn_implementation,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).to(device)

    model.eval()
    return processor, model, device, dtype


def get_step_action(step):
    action = step["action"]
    if isinstance(action, dict):
        world_vector = action["world_vector"].numpy()
        rotation_delta = action["rotation_delta"].numpy()
        open_gripper = np.array([float(action["open_gripper"].numpy())])
        return np.concatenate([world_vector, rotation_delta, open_gripper])

    action_np = action.numpy()
    return np.concatenate([action_np[:6], np.array([float(action_np[-1] > 0.5)])])


def get_step_image_and_instruction(step):
    observation = step["observation"]
    image_key = "image" if "image" in observation else "image_0"
    language_key = "natural_language_instruction"
    image = Image.fromarray(observation[image_key].numpy()).convert("RGB")
    instruction = observation[language_key].numpy().decode("utf-8")
    return image, instruction


def evaluate(args):
    processor, model, device, dtype = load_model(args)

    print(f"Loading dataset from: {args.builder_dir}")
    builder = tfds.builder_from_directory(builder_dir=args.builder_dir)
    dataset = builder.as_dataset(split=args.split, shuffle_files=False)

    if args.skip_episodes:
        dataset = dataset.skip(args.skip_episodes)

    episode = next(iter(dataset))
    steps = list(episode["steps"])
    if args.max_steps:
        steps = steps[: args.max_steps]

    first_image, task_instruction = get_step_image_and_instruction(steps[0])
    del first_image

    print("=" * 72)
    print(f"Task instruction: {task_instruction}")
    print(f"Evaluating frames: {len(steps)}")
    print(f"Checkpoint: {args.checkpoint_dir or 'base model'}")
    print("=" * 72)

    frame_results = []
    errors = []

    for idx, step in enumerate(steps, start=1):
        image, instruction = get_step_image_and_instruction(step)
        ground_truth = get_step_action(step)
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = processor(prompt, image).to(device, dtype=dtype)

        start = time.time()
        with torch.no_grad():
            predicted = model.predict_action(**inputs, unnorm_key=args.unnorm_key, do_sample=False)
        elapsed = time.time() - start

        mae = float(np.mean(np.abs(ground_truth[:6] - predicted[:6])))
        errors.append(mae)

        result = {
            "frame": idx,
            "instruction": instruction,
            "ground_truth": ground_truth.tolist(),
            "predicted": predicted.tolist() if hasattr(predicted, "tolist") else list(predicted),
            "mae_xyzrpy": mae,
            "inference_seconds": elapsed,
        }
        frame_results.append(result)

        print(f"--- Frame {idx}/{len(steps)} ---")
        print(f"Ground Truth : {np.round(ground_truth, 4)}")
        print(f"Predicted    : {np.round(predicted, 4)}")
        print(f"MAE xyz/rpy  : {mae:.6f} ({elapsed:.2f}s)")

    summary = {
        "task_instruction": task_instruction,
        "builder_dir": args.builder_dir,
        "split": args.split,
        "skip_episodes": args.skip_episodes,
        "num_frames": len(frame_results),
        "model_id": args.model_id,
        "checkpoint_dir": args.checkpoint_dir,
        "unnorm_key": args.unnorm_key,
        "average_mae_xyzrpy": float(np.mean(errors)),
        "frames": frame_results,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"openvla_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with output_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print(f"Average MAE xyz/rpy: {summary['average_mae_xyzrpy']:.6f}")
    print(f"Saved results: {output_path}")
    print("=" * 72)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate OpenVLA on a Bridge-style TFDS/RLDS episode.")
    parser.add_argument("--model-id", default="openvla/openvla-7b")
    parser.add_argument("--checkpoint-dir", default="checkpoints/my_openvla_lora")
    parser.add_argument("--builder-dir", default="gs://gresearch/robotics/bridge/0.1.0/")
    parser.add_argument("--split", default="train")
    parser.add_argument("--skip-episodes", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--unnorm-key", default="bridge_orig")
    parser.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--merge-lora", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("eval_results"))
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
