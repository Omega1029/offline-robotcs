#!/usr/bin/env python3
"""
capture.py — shared helpers for the 08C/08D/08E capture experiments.

Provides: model loading (the 90.25% epoch_003 checkpoint), building forward inputs from
LIBERO frames, frame selection (1 image / one task / N tasks), and `ActivationCapture` —
which hooks ALL backbone linears in ONE forward pass and accumulates per-input-channel
STATISTICS (max / mean-abs), never raw tensors (so memory stays ~10 MB, not ~2 GB/image).
"""
from __future__ import annotations
import json
from pathlib import Path
import h5py, numpy as np, torch
from PIL import Image

import importlib.util
PIPE = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("eval_libero", str(PIPE / "02_eval_libero.py"))
eval_libero = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(eval_libero)
import quant  # backbone_linears / skip list (same dir)

CKPT = str(PIPE / "checkpoints/openvla_oft_libero/epoch_003/hf_model")
HEAD = str(PIPE / "checkpoints/openvla_oft_libero/epoch_003/action_head.pt")
DATA = Path("/home/justin_williams1/OfflineAutonomy/offline-robotcs/datasets/libero")
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_long"]
SYS = ("A chat between a curious user and an artificial intelligence assistant. "
       "The assistant gives helpful, detailed, and polite answers to the user's questions. ")


def load_model(device):
    processor, model = eval_libero.load_model(CKPT, HEAD, device)
    model.eval()
    return processor, model


def build_inputs(processor, model, device, instr, img):
    prefix = f"{SYS}USER: What action should the robot take to {instr}? ASSISTANT: "
    pid = processor.tokenizer(prefix, add_special_tokens=True, return_tensors="pt")["input_ids"].squeeze(0)
    iid = torch.cat([pid, model.eval_action_token_ids]).unsqueeze(0).to(device)
    pv = processor.image_processor(images=img, return_tensors="pt")["pixel_values"].to(device, torch.bfloat16)
    return dict(input_ids=iid, attention_mask=torch.ones_like(iid), pixel_values=pv)


# ── frame selection ───────────────────────────────────────────────────────────
def _task_files():
    files = []
    for s in SUITES:
        d = DATA / ("libero_long/libero_10" if s == "libero_long" else s)
        files += [(s, fp) for fp in sorted(d.glob("*.hdf5"))]
    return files


def frames_one_image():
    """A single middle frame from the first libero_spatial task."""
    _, fp = _task_files()[0]
    with h5py.File(fp, "r") as f:
        instr = json.loads(f["data"].attrs["problem_info"])["language_instruction"].strip().rstrip(".")
        d = f["data"]["demo_0"]; t = d["actions"].shape[0] // 2
        return [(instr, Image.fromarray(d["obs"]["agentview_rgb"][t][::-1, ::-1]))]


def frames_one_task(suite="libero_spatial", file_idx=0, stride=1):
    """ALL timesteps of one demo (a full task trajectory)."""
    files = [f for s, f in _task_files() if s == suite]
    fp = files[file_idx]
    out = []
    with h5py.File(fp, "r") as f:
        instr = json.loads(f["data"].attrs["problem_info"])["language_instruction"].strip().rstrip(".")
        d = f["data"]["demo_0"]; T = d["actions"].shape[0]
        for t in range(0, T, stride):
            out.append((instr, Image.fromarray(d["obs"]["agentview_rgb"][t][::-1, ::-1])))
    return out


def frames_n_tasks(n, frames_per_task=8):
    """`frames_per_task` evenly-spaced frames from each of the first n task files
    (spread across suites)."""
    out = []
    for s, fp in _task_files()[:n]:
        with h5py.File(fp, "r") as f:
            instr = json.loads(f["data"].attrs["problem_info"])["language_instruction"].strip().rstrip(".")
            d = f["data"]["demo_0"]; T = d["actions"].shape[0]
            for t in np.linspace(0, T - 1, min(frames_per_task, T), dtype=int):
                out.append((instr, Image.fromarray(d["obs"]["agentview_rgb"][t][::-1, ::-1])))
    return out


# ── all-layer activation statistics (stats only, not raw) ──────────────────────
class ActivationCapture:
    """Hook every backbone linear; accumulate per-input-channel |max| and mean|.| and a
    token count, across however many forwards you run. One forward covers ALL layers."""
    def __init__(self, model):
        self.model = model
        self.absmax, self.sumabs, self.ntok = {}, {}, {}
        self._handles = []

    def _hook(self, name):
        def fn(mod, inp, out):
            x = inp[0].detach().float()
            x = x.reshape(-1, x.shape[-1])                 # (tokens, in_features)
            cmax = x.abs().amax(0)
            if name not in self.absmax:
                self.absmax[name] = cmax
                self.sumabs[name] = x.abs().sum(0)
                self.ntok[name] = x.shape[0]
            else:
                self.absmax[name] = torch.maximum(self.absmax[name], cmax)
                self.sumabs[name] += x.abs().sum(0)
                self.ntok[name] += x.shape[0]
        return fn

    def attach(self):
        for full, _, _, lin in quant.backbone_linears(self.model):
            self._handles.append(lin.register_forward_hook(self._hook(full)))
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    @torch.no_grad()
    def run(self, processor, device, frames):
        self.attach()
        for i, (instr, img) in enumerate(frames):
            self.model(**build_inputs(processor, self.model, device, instr, img))
        self.detach()
        return self

    def stats(self):
        """Per-layer dict: in_absmax (per-channel), in_meanabs (per-channel), outlier_ratio,
        n_tokens. CPU tensors, ~10 MB total."""
        out = {}
        for n in self.absmax:
            amax = self.absmax[n].cpu()
            mean = (self.sumabs[n] / max(self.ntok[n], 1)).cpu()
            out[n] = {"in_absmax": amax, "in_meanabs": mean,
                      "outlier_ratio": float(amax.max() / amax.median().clamp_min(1e-8)),
                      "n_tokens": self.ntok[n]}
        return out

    def save(self, path):
        torch.save(self.stats(), path)
        print(f"[capture] saved per-layer stats → {path}")


def summarize(stats, top=8):
    """Print the layers hardest to quantize (highest activation-outlier ratio)."""
    ranked = sorted(stats.items(), key=lambda kv: kv[1]["outlier_ratio"], reverse=True)
    print(f"\n[capture] {len(stats)} layers captured. Worst activation outliers:")
    print(f"  {'layer':50s} {'outlier_ratio':>13s} {'in_absmax':>10s}")
    for name, s in ranked[:top]:
        print(f"  {name[-50:]:50s} {s['outlier_ratio']:>13.1f} {float(s['in_absmax'].max()):>10.2f}")
    ratios = [s["outlier_ratio"] for s in stats.values()]
    print(f"  median outlier_ratio across layers = {float(np.median(ratios)):.1f}, "
          f"max = {float(np.max(ratios)):.1f}")
