#!/usr/bin/env python3
"""
Python launcher for OpenVLA LoRA fine-tuning.
Replaces train_openvla.sh — import and call run() or execute directly.
"""

import os
import subprocess
import sys
from pathlib import Path


def run(
    data_root_dir: str = "datasets/open-x-embodiment",
    dataset_name: str = "bridge_dataset",
    vla_path: str = "openvla/openvla-7b",
    run_root_dir: str = "checkpoints/bridge_lora",
    adapter_tmp_dir: str = "checkpoints/bridge_lora_adapters",
    nproc_per_node: int = 1,
    batch_size: int = 1,
    grad_accumulation_steps: int = 16,
    max_steps: int = 100,
    save_steps: int = 100,
    learning_rate: float = 5e-4,
    lora_rank: int = 32,
    lora_dropout: float = 0.0,
    image_aug: bool = False,
    use_quantization: bool = False,
    shuffle_buffer_size: int = 10_000,
    wandb_mode: str = "disabled",
):
    dataset_dir = Path(data_root_dir) / dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}\n"
            "Download/convert the RLDS dataset first, or adjust data_root_dir and dataset_name."
        )

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(Path("openvla").resolve()), env.get("PYTHONPATH", "")])
    )
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["WANDB_MODE"] = wandb_mode

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone", "--nnodes", "1",
        "--nproc-per-node", str(nproc_per_node),
        "openvla/vla-scripts/finetune.py",
        "--vla_path", vla_path,
        "--data_root_dir", data_root_dir,
        "--dataset_name", dataset_name,
        "--run_root_dir", run_root_dir,
        "--adapter_tmp_dir", adapter_tmp_dir,
        "--use_lora", "True",
        "--lora_rank", str(lora_rank),
        "--lora_dropout", str(lora_dropout),
        "--use_quantization", str(use_quantization),
        "--batch_size", str(batch_size),
        "--grad_accumulation_steps", str(grad_accumulation_steps),
        "--max_steps", str(max_steps),
        "--save_steps", str(save_steps),
        "--learning_rate", str(learning_rate),
        "--image_aug", str(image_aug),
        "--shuffle_buffer_size", str(shuffle_buffer_size),
    ]

    print("Launching fine-tuning...")
    print("Command:", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    run()
