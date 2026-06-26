#!/usr/bin/env python3
"""
09_quant_sweep.py — persistent-model quantization sweep + leaderboard.

Loads the bf16 epoch_003 OFT model ONCE, snapshots the original backbone linears, captures
a calibration buffer (per-input-channel activation stats) ONCE, then for each quant config:
restore → graft the quantized modules → eval closed-loop LIBERO → restore. Amortizing the
~25 s load + calibration across the whole sweep is what makes "try lots of combinations"
affordable. Every config is fake-quant (quant→dequant in fp), scored by LIBERO success vs
the 90.25% bf16 baseline; "size" = effective weight bits reported per config.

Each result is appended to results/quant_sweep_leaderboard.jsonl.

USAGE:
  CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl ../venv/bin/python 09_quant_sweep.py \
      --suite libero_spatial --rollouts_per_task 5 --max_tasks_per_suite 4 \
      --calib_tasks 12 --tag screen
  # restrict to specific configs:
  ... 09_quant_sweep.py --only rot_w3_had,awq_w3 ...
  # list configs without running:
  ... 09_quant_sweep.py --list
"""
import argparse, json, os, sys, time
from datetime import datetime
from pathlib import Path

PIPE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPE / "quant_expirements"))
sys.path.insert(0, str(PIPE / "openvla-oft"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import importlib.util
import torch

_spec = importlib.util.spec_from_file_location("eval_libero", str(PIPE / "02_eval_libero.py"))
eval_libero = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(eval_libero)

import quant
import quant_advanced as qa
import capture as cap_mod

CKPT = str(PIPE / "checkpoints/openvla_oft_libero/epoch_003/hf_model")
HEAD = str(PIPE / "checkpoints/openvla_oft_libero/epoch_003/action_head.pt")
STATS = str(PIPE / "checkpoints/openvla_oft_libero/epoch_003/action_stats.json")
LEADERBOARD = str(PIPE / "results/quant_sweep_leaderboard.jsonl")
BASELINE = 0.9025  # bf16 full-eval average


# ─────────────────────────── config registry ─────────────────────────────────
def build_configs():
    """label -> technique instance (each exposes .apply(model, ctx) -> effective_bits).
    The uniform baselines wrap quant.FakeQuant (apply(model) signature handled in run)."""
    cfgs = {}
    # ── baselines (naive per-channel uniform) ──
    for b in (8, 4, 3, 2):
        cfgs[f"uniform_w{b}"] = quant.FakeQuant(w_bits=b)
    cfgs["uniform_w4a8"] = quant.FakeQuant(w_bits=4, a_bits=8)
    cfgs["uniform_w4a4"] = quant.FakeQuant(w_bits=4, a_bits=4)

    # ── rotation (Hadamard/orthogonal incoherence) ──
    for b in (4, 3, 2):
        cfgs[f"rot_w{b}_had"] = qa.Rotation(w_bits=b, kind="hadamard")
    cfgs["rot_w4a8_had"] = qa.Rotation(w_bits=4, a_bits=8, kind="hadamard")
    cfgs["rot_w4a4_had"] = qa.Rotation(w_bits=4, a_bits=4, kind="hadamard")
    cfgs["rot_w3a8_had"] = qa.Rotation(w_bits=3, a_bits=8, kind="hadamard")
    cfgs["rot_w3a4_had"] = qa.Rotation(w_bits=3, a_bits=4, kind="hadamard")  # aggressive corner
    cfgs["rot_w2a8_had"] = qa.Rotation(w_bits=2, a_bits=8, kind="hadamard")  # W2 frontier probe

    # ── AWQ activation-aware scaling ──
    for b in (4, 3, 2):
        cfgs[f"awq_w{b}"] = qa.AWQScale(w_bits=b)
    cfgs["awq_w4a8"] = qa.AWQScale(w_bits=4, a_bits=8)

    # ── SVD low-rank residual ──
    for b, r in ((3, 32), (3, 64), (2, 64), (2, 128)):
        cfgs[f"svd_w{b}_r{r}"] = qa.SVDResidual(w_bits=b, rank=r)

    # ── mixed precision ──
    for t in (5.0, 4.5, 3.5):
        cfgs[f"mixed_t{t}"] = qa.MixedPrecision(target_bits=t)

    # ── GPTQ (Hessian error-feedback) and GPTQ+Hadamard ──
    for b in (4, 3, 2):
        cfgs[f"gptq_w{b}"] = qa.GPTQ(w_bits=b)
    for b in (3, 2):
        cfgs[f"gptq_w{b}_had"] = qa.GPTQ(w_bits=b, rotate=True)

    # ── transform-zoo champions @ W4A4 (one per category, for proxy→closed-loop validation) ──
    for kind in ("dct", "srht", "random_orthogonal", "butterfly", "haar", "pca", "permutation"):
        cfgs[f"tf_{kind}_w4a4"] = qa.Transform(kind=kind, w_bits=4, a_bits=4)

    return cfgs


def apply_config(tech, model, ctx):
    """Call .apply with ctx if the technique's signature accepts it, else the bare model.
    Coerce the return to effective-bits: methods return a float; the uniform FakeQuant
    baseline returns the model, so fall back to its nominal w_bits."""
    import inspect
    params = inspect.signature(tech.apply).parameters
    eff = tech.apply(model, ctx) if len(params) >= 2 else tech.apply(model)
    if not isinstance(eff, (int, float)):
        eff = float(getattr(tech, "w_bits", 0) or 0)
    return eff


# ─────────────────────────── sweep machinery ─────────────────────────────────
def snapshot_backbone(model):
    """Return a restore() closure capturing the original nn.Linear modules (so each config
    starts from clean bf16 weights; methods only find nn.Linear via backbone_linears)."""
    originals = [(parent, attr, child) for _, parent, attr, child in quant.backbone_linears(model)]
    def restore():
        for parent, attr, child in originals:
            setattr(parent, attr, child)
    return restore, len(originals)


def capture_calibration(model, processor, device, n_tasks, frames_per_task):
    """Per-input-channel mean|x| for every backbone linear over an N-task buffer. Reused by
    all calibration-based methods (AWQ, mixed). Stats only (~10 MB), not raw activations."""
    frames = cap_mod.frames_n_tasks(n_tasks, frames_per_task)
    cap = cap_mod.ActivationCapture(model).run(processor, device, frames)
    stats = cap.stats()
    return {"act_meanabs": {n: s["in_meanabs"] for n, s in stats.items()},
            "act_absmax": {n: s["in_absmax"] for n, s in stats.items()},
            "n_calib_frames": len(frames),
            # GPTQ captures its own per-layer Hessians from these forward inputs:
            "calib_frames": frames, "processor": processor, "device": device}


def run_eval(model, processor, action_mean, action_std, device, suites, rollouts, max_tasks):
    results = []
    for s in suites:
        results.extend(eval_libero.evaluate_suite(
            suite_name=s, model=model, processor=processor,
            action_mean=action_mean, action_std=action_std, device=device,
            rollouts_per_task=rollouts, max_tasks=max_tasks))
    succ = sum(r["success"] for r in results) / max(len(results), 1)
    per_task = {}
    for r in results:
        per_task.setdefault(r["task"], []).append(r["success"])
    per_task = {k: sum(v) / len(v) for k, v in per_task.items()}
    return succ, len(results), per_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--rollouts_per_task", type=int, default=5)
    ap.add_argument("--max_tasks_per_suite", type=int, default=4)
    ap.add_argument("--calib_tasks", type=int, default=12)
    ap.add_argument("--calib_frames_per_task", type=int, default=6)
    ap.add_argument("--only", default=None, help="comma-separated config labels")
    ap.add_argument("--list", action="store_true", help="list configs and exit")
    ap.add_argument("--tag", default="screen")
    args = ap.parse_args()

    configs = build_configs()
    if args.list:
        for k, v in configs.items():
            print(f"  {k:18s} {type(v).__name__}")
        return
    if args.only:
        want = [s.strip() for s in args.only.split(",")]
        configs = {k: configs[k] for k in want if k in configs}
        missing = [s for s in want if s not in build_configs()]
        if missing:
            print(f"[warn] unknown configs ignored: {missing}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    suites = (eval_libero.LIBERO_SUITES if args.suite in ("", "all", None)
              else [args.suite])

    t0 = time.time()
    print(f"[Sweep] loading bf16 model …")
    processor, model = eval_libero.load_model(CKPT, HEAD, device)
    model.eval()
    restore, n_lin = snapshot_backbone(model)
    print(f"[Sweep] snapshot {n_lin} backbone linears | load {time.time()-t0:.0f}s")

    stats = json.load(open(STATS))
    action_mean = torch.tensor(stats["mean"], device=device, dtype=torch.bfloat16)
    action_std = torch.tensor(stats["std"], device=device, dtype=torch.bfloat16)

    print(f"[Sweep] capturing calibration buffer ({args.calib_tasks} tasks) …")
    tc = time.time()
    ctx = capture_calibration(model, processor, device, args.calib_tasks, args.calib_frames_per_task)
    print(f"[Sweep] calib {ctx['n_calib_frames']} frames in {time.time()-tc:.0f}s")

    Path(LEADERBOARD).parent.mkdir(parents=True, exist_ok=True)
    for label, tech in configs.items():
        restore()
        print(f"\n{'='*70}\n[Sweep] CONFIG: {label}\n{'='*70}")
        ts = time.time()
        try:
            eff_bits = apply_config(tech, model, ctx)
            model.eval()
            succ, n_roll, per_task = run_eval(
                model, processor, action_mean, action_std, device, suites,
                args.rollouts_per_task, args.max_tasks_per_suite)
            rec = {
                "label": label, "method": type(tech).__name__,
                "w_bits": getattr(tech, "w_bits", None), "a_bits": getattr(tech, "a_bits", None),
                "effective_bits": round(float(eff_bits), 3),
                "suite": args.suite, "rollouts_per_task": args.rollouts_per_task,
                "max_tasks": args.max_tasks_per_suite, "n_rollouts": n_roll,
                "success": round(succ, 4), "delta_vs_baseline": round(succ - BASELINE, 4),
                "per_task": {k: round(v, 3) for k, v in per_task.items()},
                "tag": args.tag, "seconds": round(time.time() - ts, 1),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        except Exception as e:
            import traceback; traceback.print_exc()
            rec = {"label": label, "method": type(tech).__name__, "error": repr(e),
                   "tag": args.tag, "timestamp": datetime.now().isoformat(timespec="seconds")}
        with open(LEADERBOARD, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if "success" in rec:
            print(f"[Sweep] {label}: success={rec['success']:.1%} "
                  f"eff_bits={rec['effective_bits']} ({rec['seconds']}s)")
    restore()
    print(f"\n[Sweep] done in {(time.time()-t0)/60:.1f} min → {LEADERBOARD}")


if __name__ == "__main__":
    main()
