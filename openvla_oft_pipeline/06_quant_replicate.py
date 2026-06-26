#!/usr/bin/env python3
"""
Phase 0 scaffold for replicating QVLA (2602.03782) and QuantVLA (2602.20309) on
OpenVLA-OFT via SIMULATED (fake) quantization — validates accuracy retention vs the
bf16 baseline without custom low-bit kernels.

Design:
  - Load the bf16 OFT checkpoint exactly like 02_eval_libero.py (no bnb).
  - Replace nn.Linear modules INSIDE the Llama backbone with FakeQuantLinear, which
    quantizes→dequantizes weights (and optionally activations) on each forward.
    Vision tower, projector, lm_head, and the L1 action head stay bf16 (the skip list
    matches 03_quantize_awq.py / 02A_eval_quant_libero.py).
  - Reattach the bf16 action head and score with the SAME LIBERO rollout machinery
    imported from 02_eval_libero.py, so every method is comparable to the 90.25% baseline.

Methods (--method):
  uniform   : Phase-0 sanity. Per-output-channel symmetric weight quant at --w_bits.
              (W8 should ≈ bf16; W4 should drop a little.) Optional --a_bits activations.
  qvla      : (Phase 1, not yet implemented) per-channel action-sensitivity bit allocation.
  quantvla  : (Phase 2, not yet implemented) scale-calibrated PTQ, attention proj kept FP.

USAGE (smoke test):
  python 06_quant_replicate.py --method uniform --w_bits 8 \
      --checkpoint checkpoints/openvla_oft_libero/epoch_003/hf_model \
      --action_stats checkpoints/openvla_oft_libero/epoch_003/action_stats.json \
      --action_head checkpoints/openvla_oft_libero/epoch_003/action_head.pt \
      --rollouts_per_task 2 --max_tasks_per_suite 2 --suite libero_spatial \
      --output_csv results/quant_uniform_w8.csv
"""
import argparse, json, os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the byte-identical eval machinery + constants from the fp eval.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "eval_libero", str(Path(__file__).parent / "02_eval_libero.py")
)
eval_libero = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_libero)

OFT_ACTION_DIM = eval_libero.OFT_ACTION_DIM
OFT_CHUNK_SIZE = eval_libero.OFT_CHUNK_SIZE
# Backbone linears whose PARENT path contains any of these are kept at bf16.
SKIP_SUBSTRINGS = ["vision_backbone", "projector", "lm_head"]


# ─────────────────────────── fake-quant primitives ───────────────────────────
def fake_quant_per_channel(w: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
    """Symmetric per-output-channel weight fake-quant. `bits` is a (out_features,)
    int tensor; bits==0 prunes that channel to zero (QVLA's 0-bit case)."""
    w = w.float()
    out = torch.empty_like(w)
    # Group rows by bit-width to vectorize.
    for b in bits.unique().tolist():
        rows = bits == b
        if b == 0:
            out[rows] = 0.0
            continue
        wr = w[rows]
        qmax = 2 ** (b - 1) - 1
        scale = wr.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
        out[rows] = torch.round(wr / scale).clamp(-qmax - 1, qmax) * scale
    return out


def fake_quant_per_tensor_act(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric per-tensor activation fake-quant (used by QuantVLA-style W?A? paths)."""
    if bits is None or bits >= 16:
        return x
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax().clamp_min(1e-8) / qmax
    return torch.round(x / scale).clamp(-qmax - 1, qmax) * scale


class FakeQuantLinear(nn.Module):
    """Drop-in wrapper around an nn.Linear. Weights are fake-quantized once at
    construction (per-channel bit vector); activations optionally fake-quantized
    each forward. Stores dequantized weights in the original dtype for clean matmul."""

    def __init__(self, linear: nn.Linear, w_bits, a_bits=None):
        super().__init__()
        out_f = linear.out_features
        if isinstance(w_bits, int):
            w_bits = torch.full((out_f,), w_bits, dtype=torch.int32)
        dq = fake_quant_per_channel(linear.weight.data, w_bits)
        self.register_buffer("weight", dq.to(linear.weight.dtype))
        self.bias = linear.bias
        self.a_bits = a_bits
        self.w_bits_mean = float(w_bits.float().mean())  # for reporting

    def forward(self, x):
        if self.a_bits is not None:
            x = fake_quant_per_tensor_act(x.float(), self.a_bits).to(self.weight.dtype)
        return F.linear(x, self.weight, self.bias)


# ─────────────────────────── graft into backbone ─────────────────────────────
def _target_linears(model):
    """Yield (parent_module, attr_name, full_name, linear) for every nn.Linear in
    the Llama backbone except the skip list."""
    lm = model.language_model
    for name, module in lm.named_modules():
        for attr, child in list(module.__dict__.get("_modules", {}).items()):
            if isinstance(child, nn.Linear):
                full = f"language_model.{name}.{attr}".replace("..", ".")
                if any(s in full for s in SKIP_SUBSTRINGS):
                    continue
                yield module, attr, full, child


def graft_fakequant(model, bits_for, a_bits=None):
    """Replace backbone linears with FakeQuantLinear. `bits_for(full_name, linear)`
    returns either an int or a (out_features,) bit tensor for that layer; return None
    to leave a layer bf16 (e.g. QuantVLA keeps attention projections FP)."""
    n_q, n_skip, total_bits, total_w = 0, 0, 0.0, 0
    for parent, attr, full, lin in list(_target_linears(model)):
        b = bits_for(full, lin)
        if b is None:
            n_skip += 1
            continue
        fq = FakeQuantLinear(lin, b, a_bits=a_bits).to(lin.weight.device)
        setattr(parent, attr, fq)
        n_q += 1
        total_bits += fq.w_bits_mean * lin.weight.numel()
        total_w += lin.weight.numel()
    eff = total_bits / max(total_w, 1)
    print(f"[Graft] quantized {n_q} backbone linears, kept {n_skip} FP "
          f"(+ vision/projector/lm_head/action_head bf16). Effective W-bits={eff:.2f}")
    return eff


# ─────────────────────────── method registry ─────────────────────────────────
def method_uniform(args, model, processor, device):
    """Phase-0 sanity: every backbone linear at --w_bits, per-channel symmetric."""
    return (lambda full, lin: args.w_bits), args.a_bits


def method_qvla(args, model, processor, device):
    """Phase 1: QVLA per-channel action-sensitivity bit allocation (see qvla.py)."""
    import qvla
    return qvla.build_qvla_bits_for(model, processor, args, device)


METHODS = {
    "uniform": method_uniform,
    "qvla": method_qvla,            # Phase 1
    # "quantvla": method_quantvla,  # Phase 2
}


def load_bf16_model(checkpoint, action_head_path, device):
    """Load OFT bf16 model + reattach action head (mirrors 02_eval_libero.load_model)."""
    return eval_libero.load_model(checkpoint, action_head_path, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(METHODS.keys()))
    ap.add_argument("--w_bits", type=int, default=8)
    ap.add_argument("--a_bits", type=int, default=None, help="activation bits (None=FP)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--action_head", required=True)
    ap.add_argument("--action_stats", required=True)
    ap.add_argument("--suite", default=None)
    ap.add_argument("--rollouts_per_task", type=int, default=10)
    ap.add_argument("--max_tasks_per_suite", type=int, default=None)
    ap.add_argument("--output_csv", default="results/quant_replicate.csv")
    # QVLA (Phase 1) knobs
    ap.add_argument("--data_root", default="/home/justin_williams1/OfflineAutonomy/offline-robotcs/datasets/libero",
                    help="LIBERO root for QVLA calibration")
    ap.add_argument("--qvla_target_bits", type=float, default=4.0)
    ap.add_argument("--qvla_candidate_bits", default="0,2,3,4,8")
    ap.add_argument("--qvla_calib_samples", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Quant] method={args.method} W{args.w_bits}"
          f"{'A'+str(args.a_bits) if args.a_bits else ''} device={device}")

    processor, model = load_bf16_model(args.checkpoint, args.action_head, device)

    bits_for, a_bits = METHODS[args.method](args, model, processor, device)
    eff_bits = graft_fakequant(model, bits_for, a_bits=a_bits)
    model.eval()

    stats = json.load(open(args.action_stats))
    action_mean = torch.tensor(stats["mean"], device=device, dtype=torch.bfloat16)
    action_std = torch.tensor(stats["std"], device=device, dtype=torch.bfloat16)

    suites = [args.suite] if args.suite else eval_libero.LIBERO_SUITES
    all_results = []
    for suite_name in suites:
        print(f"\n[Quant] Suite: {suite_name}")
        results = eval_libero.evaluate_suite(
            suite_name=suite_name, model=model, processor=processor,
            action_mean=action_mean, action_std=action_std, device=device,
            rollouts_per_task=args.rollouts_per_task, max_tasks=args.max_tasks_per_suite,
        )
        sr = sum(r["success"] for r in results) / len(results)
        print(f"[Quant] {suite_name} SUCCESS RATE: {sr:.1%}")
        all_results.extend(results)

    overall = sum(r["success"] for r in all_results) / len(all_results)
    print(f"\n[Quant] OVERALL ({args.method} W{args.w_bits}"
          f"{'A'+str(args.a_bits) if args.a_bits else ''}) = {overall:.1%} "
          f"| effective W-bits={eff_bits:.2f}")

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    import csv
    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        w.writeheader(); w.writerows(all_results)
    summary = {
        "method": args.method, "w_bits": args.w_bits, "a_bits": args.a_bits,
        "effective_w_bits": eff_bits,
        "suite_success_rates": {
            s: sum(r["success"] for r in all_results if r["suite"] == s)
               / max(1, len([r for r in all_results if r["suite"] == s]))
            for s in suites
        },
        "average_success_rate": overall,
    }
    json.dump(summary, open(args.output_csv.replace(".csv", ".summary.json"), "w"), indent=2)
    print(f"[Quant] wrote {args.output_csv}")


if __name__ == "__main__":
    main()
