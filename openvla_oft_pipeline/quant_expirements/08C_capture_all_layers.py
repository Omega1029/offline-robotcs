#!/usr/bin/env python3
"""
08C: load the 90.25% model and capture per-channel activation statistics across ALL
backbone layers from ONE image (single forward, all 224 linears hooked at once).

Output: prints the layers with the worst activation outliers (hardest to quantize) and
saves per-layer stats to quant_expirements/stats/all_layers_one_image.pt.

Run:  CUDA_VISIBLE_DEVICES=3 MUJOCO_GL=egl python quant_expirements/08C_capture_all_layers.py
"""
from pathlib import Path
import torch
import capture

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    processor, model = capture.load_model(device)
    frames = capture.frames_one_image()
    print(f"[08C] {len(frames)} image, all backbone layers")
    cap = capture.ActivationCapture(model).run(processor, device, frames)
    stats = cap.stats()
    capture.summarize(stats)
    out = Path(__file__).parent / "stats"; out.mkdir(exist_ok=True)
    cap.save(out / "all_layers_one_image.pt")

if __name__ == "__main__":
    main()
