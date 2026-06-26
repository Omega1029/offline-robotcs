#!/usr/bin/env python3
"""
08E: capture per-channel activation statistics across ALL backbone layers over N TASKS
(spread across the four LIBERO suites). This is the proper calibration buffer — broad
coverage of instructions/scenes gives the most representative per-channel activation
ranges for setting quantization scales.

Output: stats → quant_expirements/stats/all_layers_<N>tasks.pt
        (consume these in quant.py techniques to set activation scales.)

Run:  CUDA_VISIBLE_DEVICES=3 MUJOCO_GL=egl python quant_expirements/08E_capture_n_tasks.py \
          --n 16 [--frames_per_task 8]
"""
import argparse
from pathlib import Path
import torch
import capture

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16, help="number of task files to sample")
    ap.add_argument("--frames_per_task", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    processor, model = capture.load_model(device)
    frames = capture.frames_n_tasks(args.n, args.frames_per_task)
    print(f"[08E] {args.n} tasks × {args.frames_per_task} frames = {len(frames)} total, all backbone layers")
    cap = capture.ActivationCapture(model).run(processor, device, frames)
    stats = cap.stats()
    capture.summarize(stats)
    out = Path(__file__).parent / "stats"; out.mkdir(exist_ok=True)
    cap.save(out / f"all_layers_{args.n}tasks.pt")

if __name__ == "__main__":
    main()
