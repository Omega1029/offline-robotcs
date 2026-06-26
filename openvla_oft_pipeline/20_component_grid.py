#!/usr/bin/env python3
"""
20_component_grid.py — Phase-1 COMPONENT-WISE quantization grid (mono-cam epoch_003).

Treats OpenVLA-OFT as three independently-quantizable components and grids precision
across them:
    vision_backbone  (DINOv2 + SigLIP fused tower)   → INT8 / FP8(E4M3,E5M2)
    language_model   (Llama-2-7B backbone)           → INT4 + DCT rotation (W4A4) etc.
    action_head      (L1RegressionActionHead MLP)     → channel-wise INT8/INT4/INT3

Like 09_quant_sweep.py it loads the bf16 model ONCE, snapshots all three component groups,
then for each config: restore → graft quantized modules into the chosen components → eval
closed-loop LIBERO → restore. All fake-quant (quant→dequant in fp); success vs the bf16
baseline. Size is estimated per-component from exact param counts × effective bits.

Phase-3 threshold: a config is flagged NOVELTY_CANDIDATE if it achieves >70% memory
reduction while retaining >95% of the bf16 success rate.

USAGE:
  CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl ../venv/bin/python 20_component_grid.py \
      --suite libero_spatial --rollouts_per_task 5 --max_tasks_per_suite 4 --tag grid
  ... --list            # list configs and exit
  ... --only head_int4,combo_aggressive
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

CKPT = str(PIPE / "checkpoints/openvla_oft_libero/epoch_003/hf_model")
HEAD = str(PIPE / "checkpoints/openvla_oft_libero/epoch_003/action_head.pt")
STATS = str(PIPE / "checkpoints/openvla_oft_libero/epoch_003/action_stats.json")
LEADERBOARD = str(PIPE / "results/component_grid_leaderboard.jsonl")
BASELINE = 0.9025          # bf16 full-eval average
RETAIN_THRESH = 0.95       # >95% of baseline success
MEM_REDUCE_THRESH = 0.70   # >70% memory reduction


def build_configs():
    """label -> quant.ComponentQuant. None = bf16. Backbone DCT-W4A4 is the established
    champion (reused as backbone_tech). Grid = component isolation + combined candidates."""
    def dct(w=4, a=4):
        return qa.Transform(kind="dct", w_bits=w, a_bits=a)
    cfgs = {}
    # ── true bf16 baseline (no-op graft) — for honest per-suite retention ──
    cfgs["bf16_baseline"] = quant.ComponentQuant()
    # ── Phase 1a: component sensitivity (isolate one, others bf16) ──
    cfgs["head_int8"]       = quant.ComponentQuant(head=8)
    cfgs["head_int4"]       = quant.ComponentQuant(head=4)
    cfgs["head_int3"]       = quant.ComponentQuant(head=3)
    cfgs["vision_int8"]     = quant.ComponentQuant(vision=8)
    cfgs["vision_fp8"]      = quant.ComponentQuant(vision="fp8")
    cfgs["vision_fp8_e5m2"] = quant.ComponentQuant(vision="fp8_e5m2")
    cfgs["backbone_w4_dct"]  = quant.ComponentQuant(backbone_tech=dct(4, None))
    cfgs["backbone_w4a4_dct"]= quant.ComponentQuant(backbone_tech=dct(4, 4))
    # ── Phase 1b: combined deployment candidates ──
    cfgs["combo_full_int8"]      = quant.ComponentQuant(vision=8, backbone=8, head=8)
    cfgs["combo_v8_b4dct_h8"]    = quant.ComponentQuant(vision=8, head=8, backbone_tech=dct(4, 4))
    cfgs["combo_vfp8_b4dct_h8"]  = quant.ComponentQuant(vision="fp8", head=8, backbone_tech=dct(4, 4))
    cfgs["combo_v8_b4dct_h4"]    = quant.ComponentQuant(vision=8, head=4, backbone_tech=dct(4, 4))
    cfgs["combo_aggressive"]     = quant.ComponentQuant(vision="fp8", head=4, backbone_tech=dct(4, 4))
    # ── Phase 1c: push past 70% size — shrink the backbone (head stays INT8) ──
    cfgs["combo_v8_b3a8dct_h8"]  = quant.ComponentQuant(vision=8, head=8, backbone_tech=dct(3, 8))
    cfgs["combo_v8_b3a4dct_h8"]  = quant.ComponentQuant(vision=8, head=8, backbone_tech=dct(3, 4))
    cfgs["combo_vfp8_b3a8dct_h8"]= quant.ComponentQuant(vision="fp8", head=8, backbone_tech=dct(3, 8))
    return cfgs


# ─────────────────────────── size accounting ─────────────────────────────────
def group_param_counts(model):
    """Exact #params in each quantizable component + the bf16-frozen remainder."""
    counts = {
        "vision":   sum(l.weight.numel() for *_, l in quant.vision_linears(model)),
        "backbone": sum(l.weight.numel() for *_, l in quant.backbone_linears(model)),
        "head":     sum(l.weight.numel() for *_, l in quant.action_head_linears(model)),
    }
    total = sum(p.numel() for p in model.parameters())
    counts["frozen_bf16"] = total - sum(counts.values())  # projector, lm_head, embeds, norms
    counts["total"] = total
    return counts


def _eff_bits(spec, default_int=16):
    """Effective storage bits for a component spec (None=16, int=bits, fp8=8)."""
    if spec is None:
        return 16
    if isinstance(spec, str):
        return 8 if spec.startswith("fp8") else default_int
    return int(spec)


def estimate_size(tech, counts):
    """Estimate model bytes under a ComponentQuant config; vision/backbone/head per their
    spec, everything else bf16 (16-bit). Returns (gb, reduction_vs_bf16)."""
    vb = _eff_bits(tech.vision)
    hb = _eff_bits(tech.head)
    if tech.backbone_tech is not None:
        bb = int(getattr(tech.backbone_tech, "w_bits", 4) or 4)
    else:
        bb = _eff_bits(tech.backbone)
    bits = (counts["vision"] * vb + counts["backbone"] * bb +
            counts["head"] * hb + counts["frozen_bf16"] * 16)
    gb = bits / 8 / 1e9
    bf16_gb = counts["total"] * 16 / 8 / 1e9
    return gb, 1.0 - gb / bf16_gb


# ─────────────────────────── sweep machinery ─────────────────────────────────
def snapshot_all(model):
    """Capture original modules for ALL three component groups so each config starts clean."""
    originals = []
    for enum in (quant.backbone_linears, quant.vision_linears, quant.action_head_linears):
        for _full, parent, attr, child in enum(model):
            originals.append((parent, attr, child))
    def restore():
        for parent, attr, child in originals:
            setattr(parent, attr, child)
    return restore, len(originals)


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
    ap.add_argument("--only", default=None, help="comma-separated config labels")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--tag", default="grid")
    args = ap.parse_args()

    configs = build_configs()
    if args.list:
        for k, v in configs.items():
            print(f"  {k:22s} vision={v.vision} backbone="
                  f"{'dct'+str(getattr(v.backbone_tech,'w_bits','')) if v.backbone_tech else v.backbone}"
                  f" head={v.head}")
        return
    if args.only:
        want = [s.strip() for s in args.only.split(",")]
        configs = {k: configs[k] for k in want if k in configs}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    suites = (eval_libero.LIBERO_SUITES if args.suite in ("", "all", None) else [args.suite])

    t0 = time.time()
    print("[Grid] loading bf16 mono-cam model …")
    processor, model = eval_libero.load_model(CKPT, HEAD, device)
    model.eval()
    counts = group_param_counts(model)
    print(f"[Grid] params: vision={counts['vision']/1e6:.0f}M  "
          f"backbone={counts['backbone']/1e6:.0f}M  head={counts['head']/1e6:.2f}M  "
          f"frozen={counts['frozen_bf16']/1e6:.0f}M  total={counts['total']/1e9:.2f}B")
    restore, n_lin = snapshot_all(model)
    print(f"[Grid] snapshot {n_lin} linears across 3 components | load {time.time()-t0:.0f}s")

    stats = json.load(open(STATS))
    action_mean = torch.tensor(stats["mean"], device=device, dtype=torch.bfloat16)
    action_std = torch.tensor(stats["std"], device=device, dtype=torch.bfloat16)

    Path(LEADERBOARD).parent.mkdir(parents=True, exist_ok=True)
    for label, tech in configs.items():
        restore()
        print(f"\n{'='*70}\n[Grid] CONFIG: {label}\n{'='*70}")
        ts = time.time()
        try:
            tech.apply(model, None)
            model.eval()
            gb, reduction = estimate_size(tech, counts)
            succ, n_roll, per_task = run_eval(
                model, processor, action_mean, action_std, device, suites,
                args.rollouts_per_task, args.max_tasks_per_suite)
            retention = succ / BASELINE if BASELINE else 0.0
            flag = (reduction > MEM_REDUCE_THRESH and retention > RETAIN_THRESH)
            rec = {
                "label": label, "vision": tech.vision,
                "backbone": ("dct_w%s" % getattr(tech.backbone_tech, "w_bits", "")
                             if tech.backbone_tech else tech.backbone),
                "head": tech.head,
                "suite": args.suite, "n_rollouts": n_roll,
                "success": round(succ, 4), "retention": round(retention, 4),
                "est_size_gb": round(gb, 3), "mem_reduction": round(reduction, 4),
                "novelty_candidate": bool(flag),
                "per_task": {k: round(v, 3) for k, v in per_task.items()},
                "tag": args.tag, "seconds": round(time.time() - ts, 1),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            if flag:
                print(f"  ★★★ NOVELTY_CANDIDATE: {reduction:.0%} smaller, "
                      f"{retention:.1%} retention — flag for literature check ★★★")
        except Exception as e:
            import traceback; traceback.print_exc()
            rec = {"label": label, "error": repr(e), "tag": args.tag,
                   "timestamp": datetime.now().isoformat(timespec="seconds")}
        with open(LEADERBOARD, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if "success" in rec:
            print(f"[Grid] {label}: success={rec['success']:.1%} "
                  f"size={rec['est_size_gb']}GB ({rec['mem_reduction']:.0%}↓) "
                  f"retention={rec['retention']:.1%} ({rec['seconds']}s)")
    restore()
    print(f"\n[Grid] done in {(time.time()-t0)/60:.1f} min → {LEADERBOARD}")


if __name__ == "__main__":
    main()
