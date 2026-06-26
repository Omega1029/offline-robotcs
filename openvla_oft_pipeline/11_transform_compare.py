#!/usr/bin/env python3
"""
11_transform_compare.py — try ALL the rotations/transforms and rank them by how well each
tames activations for low-bit quant, WITHOUT running rollouts. Closed-loop eval is slow
(~minutes/config); the mechanism we proved (rotation kills activation outliers → A4 works)
is measurable offline in seconds, so this screens the whole zoo fast. Promising transforms
then get rollout-confirmed via 09_quant_sweep.py.

Protocol:
  1. Capture up to MAX_TOK input tokens for every backbone linear over a calibration buffer
     (one forward pass, stored fp16 on CPU).
  2. For each transform kind (rotations.STARRED first, then rotations.ALL), per layer build
     the transform M (data-independent ones cached per input-dim; pca/zca from the layer's
     input covariance; polar from the layer weight), apply z = M x, and measure:
        outlier ratio   = max_channel|·| / median_channel|·|   (pre vs post)
        A4 per-tensor rel-error                                  (pre vs post — the op that
                                                                  collapses at A4 naively)
  3. Rank transforms by median post-transform A4 error (lower = better) and report.

Saves results/transform_compare.json.

Run:  CUDA_VISIBLE_DEVICES=3 MUJOCO_GL=egl ../venv/bin/python 11_transform_compare.py \
          --set starred   [--n_tasks 8 --max_tok 1024]
      ... --set all       # the full zoo
"""
import argparse, json, os, sys, time
from pathlib import Path

PIPE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPE / "quant_expirements"))
sys.path.insert(0, str(PIPE / "openvla-oft"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

import capture as cap_mod
import quant
import rotations as R
from quant import FakeQuant


def a4_relerr(x):
    """INT4 per-tensor (single-scale) activation quant relative error — the failure op."""
    xq = FakeQuant.quant_act(x.float(), 4, mode="per_tensor")
    return float((xq - x.float()).abs().mean() / (x.float().abs().mean() + 1e-9))


def outlier_ratio(x):
    """max-channel / median-channel absmax over tokens."""
    a = x.abs().amax(0)
    return float(a.max() / a.median().clamp_min(1e-8))


class ActSampler:
    """Hook every backbone linear; keep up to max_tok input tokens (fp16, CPU) per layer."""
    def __init__(self, model, max_tok):
        self.model, self.max_tok = model, max_tok
        self.X, self.W, self.dim = {}, {}, {}
        self._handles = []

    def _hook(self, name, lin):
        def fn(mod, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1])
            if name not in self.X:
                self.X[name] = x[: self.max_tok].half().cpu()
                self.W[name] = lin.weight.detach()
                self.dim[name] = lin.in_features
            elif self.X[name].shape[0] < self.max_tok:
                need = self.max_tok - self.X[name].shape[0]
                self.X[name] = torch.cat([self.X[name], x[:need].half().cpu()], 0)
        return fn

    @torch.no_grad()
    def run(self, processor, device, frames):
        for full, _, _, lin in quant.backbone_linears(self.model):
            self._handles.append(lin.register_forward_hook(self._hook(full, lin)))
        for instr, img in frames:
            self.model(**cap_mod.build_inputs(processor, self.model, device, instr, img))
        for h in self._handles:
            h.remove()
        return self


def evaluate_kind(kind, sampler, device):
    """Median pre/post outlier ratio and A4 error across layers for one transform."""
    cache = {}                                   # per-dim M for data-independent kinds
    o_pre, o_post, e_pre, e_post = [], [], [], []
    fellbacks = 0
    for name, Xh in sampler.X.items():
        X = Xh.to(device).float()
        d = sampler.dim[name]
        cov = X.t() @ X if kind in R.DATA_DEP else None
        W = sampler.W[name].to(device) if kind in R.WEIGHT_DEP else None
        if kind in R.DATA_DEP or kind in R.WEIGHT_DEP:
            M, Minv, fb = R.build_transform(d, kind, device, cov=cov, W=W)
        else:
            if d not in cache:
                cache[d] = R.build_transform(d, kind, device)
            M, Minv, fb = cache[d]
        fellbacks += int(fb)
        z = X @ M.t()
        o_pre.append(outlier_ratio(X)); o_post.append(outlier_ratio(z))
        e_pre.append(a4_relerr(X)); e_post.append(a4_relerr(z))
        del X, z, cov, W
    torch.cuda.empty_cache()
    return {
        "outlier_pre": float(np.median(o_pre)), "outlier_post": float(np.median(o_post)),
        "a4err_pre_pct": float(np.median(e_pre) * 100), "a4err_post_pct": float(np.median(e_post) * 100),
        "fellback_layers": fellbacks, "n_layers": len(o_pre),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["starred", "all"], default="starred")
    ap.add_argument("--n_tasks", type=int, default=8)
    ap.add_argument("--frames_per_task", type=int, default=4)
    ap.add_argument("--max_tok", type=int, default=1024)
    ap.add_argument("--out", default="results/transform_compare.json")
    args = ap.parse_args()

    kinds = R.STARRED if args.set == "starred" else R.ALL
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    processor, model = cap_mod.load_model(device)
    frames = cap_mod.frames_n_tasks(args.n_tasks, args.frames_per_task)
    print(f"[Compare] capturing ≤{args.max_tok} tok/layer over {len(frames)} frames …")
    t0 = time.time()
    sampler = ActSampler(model, args.max_tok).run(processor, device, frames)
    print(f"[Compare] captured {len(sampler.X)} layers in {time.time()-t0:.0f}s; "
          f"evaluating {len(kinds)} transforms: {kinds}")

    results = {}
    for k in kinds:
        tk = time.time()
        try:
            results[k] = evaluate_kind(k, sampler, device)
            results[k]["seconds"] = round(time.time() - tk, 1)
            r = results[k]
            print(f"  {k:18s} outlier {r['outlier_pre']:7.1f}→{r['outlier_post']:5.2f}  "
                  f"A4err {r['a4err_pre_pct']:5.1f}%→{r['a4err_post_pct']:5.1f}%  "
                  f"({'fb '+str(r['fellback_layers']) if r['fellback_layers'] else 'no-fb':>6s}, {r['seconds']}s)")
        except Exception as e:
            import traceback; traceback.print_exc()
            results[k] = {"error": repr(e)}

    # ── ranked summary ────────────────────────────────────────────────────────
    ok = {k: v for k, v in results.items() if "a4err_post_pct" in v}
    ranked = sorted(ok.items(), key=lambda kv: kv[1]["a4err_post_pct"])
    print("\n" + "=" * 74)
    print(f"TRANSFORM RANKING ({args.set}) — by median A4 per-tensor error after transform")
    print("=" * 74)
    print(f"  {'rank':4s} {'transform':18s} {'outlier_post':>12s} {'A4err_post':>11s} {'A4err_pre→post':>16s}")
    for i, (k, v) in enumerate(ranked, 1):
        print(f"  {i:<4d} {k:18s} {v['outlier_post']:12.2f} {v['a4err_post_pct']:10.1f}% "
              f"{v['a4err_pre_pct']:7.1f}→{v['a4err_post_pct']:.1f}%")
    print("=" * 74)
    print("Lower A4err_post = better activation conditioning for 4-bit quant. (Naive/identity\n"
          "≈ 90%+; the rotations that drop it toward ~25% are the ones that make W*A4 viable.)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"set": args.set, "n_frames": len(frames), "max_tok": args.max_tok,
               "results": results}, open(args.out, "w"), indent=2)
    print(f"[Compare] saved → {args.out}")


if __name__ == "__main__":
    main()
