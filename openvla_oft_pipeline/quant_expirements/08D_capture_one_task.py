#!/usr/bin/env python3
"""
08D: capture per-channel activation statistics across ALL backbone layers over ONE FULL
TASK — every timestep of one demo trajectory (≈100-150 forwards). More frames → tighter
per-channel max/mean estimates than a single image (08C).

Output: stats → quant_expirements/stats/all_layers_one_task.pt

Run:  CUDA_VISIBLE_DEVICES=3 MUJOCO_GL=egl python quant_expirements/08D_capture_one_task.py \
          [--suite libero_spatial] [--file_idx 0] [--stride 1]
"""
import argparse
from pathlib import Path
import torch
import capture

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--file_idx", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1, help="subsample timesteps (1 = every frame)")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    processor, model = capture.load_model(device)
    frames = capture.frames_one_task(args.suite, args.file_idx, args.stride)
    print(f"[08D] one task ({args.suite} #{args.file_idx}): {len(frames)} frames, all backbone layers")
    cap = capture.ActivationCapture(model).run(processor, device, frames)
    stats = cap.stats()
    capture.summarize(stats)
    out = Path(__file__).parent / "stats"; out.mkdir(exist_ok=True)
    cap.save(out / "all_layers_one_task.pt")

if __name__ == "__main__":
    main()
