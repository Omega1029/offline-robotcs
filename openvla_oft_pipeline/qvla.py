#!/usr/bin/env python3
"""
QVLA (arXiv 2602.03782, "Not All Channels Are Equal") — per-channel, action-centric
bit allocation for the OpenVLA-OFT Llama backbone. Implemented from the paper (no
official code). Plugs into 06_quant_replicate.py's graft as a `bits_for` provider.

Faithful-but-tractable replication of the core idea:
  The paper measures "final action-space sensitivity when quantizing each individual
  channel to various bit-widths." A literal per-channel forward sweep is intractable
  for a 7B backbone, so we use a first-order surrogate:

    sensitivity_c(b) = g_c * quant_err_c(b) * in_scale_layer

  where
    g_c          = E_calib | d(action)/d(y_c) |   (y_c = layer output channel c),
                   from ONE backward pass of L = |predicted_action_chunk|.sum()
    quant_err_c(b) = || W_c - Q_b(W_c) ||_2  over the input dim (b=0 ⇒ full weight, pruned)
    in_scale     = E_calib mean|input| to that layer (captured by a forward hook)

  Bits are then allocated per output channel by a Lagrangian sweep over candidate
  bit-widths to meet a target average bit-width, folding 0-bit (pruning) into the same
  objective — exactly QVLA's "unify quantization and pruning" framing.

This module exposes build_qvla_bits_for(model, processor, args, device) -> (bits_for, a_bits).
"""
import h5py
import numpy as np
import torch
import torch.nn as nn

# Reuse the fp eval machinery loaded by 06 (constants + prompt/image conventions).
SYS_MSG = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions. "
)
SKIP_SUBSTRINGS = ["vision_backbone", "projector", "lm_head"]


# ───────────────────────── calibration (instruction + image → action inputs) ─────
def _read_instruction(f):
    import json
    return json.loads(f["data"].attrs["problem_info"])["language_instruction"].strip().rstrip(".")


def load_calib_inputs(data_root, processor, model, device, suites, n_samples=64):
    """Build forward inputs that actually produce an action chunk (prefix + appended
    placeholder action tokens), mirroring 02_eval_libero.run_rollout."""
    from PIL import Image
    from pathlib import Path
    per_suite = max(1, n_samples // len(suites))
    samples = []
    for suite in suites:
        files = sorted((Path(data_root) / suite).glob("*.hdf5"))
        cnt = 0
        for fp in files:
            if cnt >= per_suite:
                break
            with h5py.File(fp, "r") as f:
                instr = _read_instruction(f)
                prefix = f"{SYS_MSG}USER: What action should the robot take to {instr}? ASSISTANT: "
                prefix_ids = processor.tokenizer(prefix, add_special_tokens=True,
                                                 return_tensors="pt")["input_ids"].squeeze(0)
                input_ids = torch.cat([prefix_ids, model.eval_action_token_ids]).unsqueeze(0)
                for dk in list(f["data"].keys()):
                    if cnt >= per_suite:
                        break
                    T = f["data"][dk]["actions"].shape[0]
                    for t in np.linspace(0, T - 1, min(3, T), dtype=int):
                        img = Image.fromarray(f["data"][dk]["obs"]["agentview_rgb"][t][::-1, ::-1])
                        pv = processor.image_processor(images=img, return_tensors="pt")["pixel_values"]
                        samples.append({
                            "input_ids": input_ids.to(device),
                            "attention_mask": torch.ones_like(input_ids).to(device),
                            "pixel_values": pv.to(device, dtype=torch.bfloat16),
                        })
                        cnt += 1
                        if cnt >= per_suite:
                            break
    print(f"[QVLA] {len(samples)} calibration samples")
    return samples


def _predict_action(model, batch, action_len):
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"], output_hidden_states=True)
    action_hidden = out.hidden_states[-1][:, -action_len:, :]
    return model.action_head.predict_action(action_hidden)  # (1, chunk, 7), normalized


# ───────────────────────── sensitivity profiling ─────────────────────────────
def _target_linears(model):
    lm = model.language_model
    for name, module in lm.named_modules():
        for attr, child in list(module.__dict__.get("_modules", {}).items()):
            if isinstance(child, nn.Linear):
                full = f"language_model.{name}.{attr}".replace("..", ".")
                if any(s in full for s in SKIP_SUBSTRINGS):
                    continue
                yield full, child


def profile_sensitivity(model, processor, calib, action_len):
    """Return per-layer dict: {full_name: {"g": (out,), "in_scale": float, "lin": Linear}}.
    g_c = E|d(|action|.sum)/d(y_c)|; in_scale = E mean|input|."""
    layers = dict(_target_linears(model))
    in_scale = {n: 0.0 for n in layers}
    g_accum = {n: torch.zeros(l.out_features, device=l.weight.device) for n, l in layers.items()}

    # Forward hooks: capture mean|input| and tag the output tensor to grab its grad.
    handles, out_grads = [], {}
    def mk_fwd(name):
        def hook(mod, inp, out):
            in_scale[name] += inp[0].detach().abs().mean().item()
            out.requires_grad_(True)
            out.retain_grad()
            out_grads[name] = out
        return hook
    for n, l in layers.items():
        handles.append(l.register_forward_hook(mk_fwd(n)))

    n_used = 0
    for b in calib:
        out_grads.clear()
        action = _predict_action(model, b, action_len)
        loss = action.float().abs().sum()
        model.zero_grad(set_to_none=True)
        loss.backward()
        for n in layers:
            g = out_grads.get(n)
            if g is not None and g.grad is not None:
                # grad shape (1, seq, out) → per-output-channel mean |grad|
                g_accum[n] += g.grad.detach().abs().reshape(-1, g.shape[-1]).mean(0)
        n_used += 1
    for h in handles:
        h.remove()

    sens = {}
    for n, l in layers.items():
        sens[n] = {
            "g": (g_accum[n] / max(n_used, 1)).float().cpu(),
            "in_scale": in_scale[n] / max(n_used, 1),
            "lin": l,
        }
    print(f"[QVLA] profiled {len(sens)} layers over {n_used} samples")
    return sens


def _quant_err_per_channel(W, b):
    """L2 quant error per output channel at bit b (b=0 ⇒ full magnitude = pruned)."""
    W = W.float()
    if b == 0:
        return W.norm(dim=1)  # pruning error = ||W_c||
    qmax = 2 ** (b - 1) - 1
    scale = W.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
    Wq = torch.round(W / scale).clamp(-qmax - 1, qmax) * scale
    return (W - Wq).norm(dim=1)


# ───────────────────────── Lagrangian bit allocation ─────────────────────────
def allocate_bits(sens, candidate_bits, target_avg_bits):
    """Per-channel bit allocation minimizing total action-sensitivity at the target
    average bit-width. cost_c(b) = g_c * quant_err_c(b) * in_scale.

    Cost tables are built ONCE (quant error computed on the weight's device, then the
    small per-channel cost vectors moved to CPU) and concatenated into one flat
    (N_total_channels, n_cand) tensor. The Lagrangian binary search is then fully
    vectorized on CPU — no per-layer Python loop or repeated host/device syncs."""
    names, costs, spans = [], [], []
    cursor = 0
    for n, d in sens.items():
        W = d["lin"].weight.data
        g = d["g"].to(W.device)
        cols = [(g * _quant_err_per_channel(W, b) * d["in_scale"]) for b in candidate_bits]
        costs.append(torch.stack(cols, dim=1).float().cpu())  # (out, n_cand) → CPU
        names.append(n)
        spans.append((cursor, cursor + W.shape[0]))
        cursor += W.shape[0]
    cost = torch.cat(costs, dim=0)                       # (N_total, n_cand) on CPU
    cand = torch.tensor(candidate_bits, dtype=torch.float32)

    def choose(lam):
        idx = (cost + lam * cand).argmin(dim=1)          # vectorized over all channels
        return cand[idx].mean().item(), idx

    lo, hi = 0.0, 1e6
    for _ in range(50):
        mid = (lo + hi) / 2
        avg, _ = choose(mid)
        if avg > target_avg_bits:                        # higher λ → fewer bits
            lo = mid
        else:
            hi = mid
    final_avg, idx = choose(hi)

    bits = {}
    for n, (s, e) in zip(names, spans):
        sel = idx[s:e]
        bits[n] = torch.tensor([candidate_bits[i] for i in sel.tolist()], dtype=torch.int32)
    # quick distribution print
    from collections import Counter
    dist = Counter(candidate_bits[i] for i in idx.tolist())
    print(f"[QVLA] allocated bits, achieved average = {final_avg:.2f} (target {target_avg_bits}) "
          f"| bit distribution {dict(sorted(dist.items()))}")
    return bits


def build_qvla_bits_for(model, processor, args, device):
    candidate_bits = [int(b) for b in args.qvla_candidate_bits.split(",")]
    calib = load_calib_inputs(args.data_root, processor, model, device,
                              suites=_suites(args), n_samples=args.qvla_calib_samples)
    action_len = model.eval_action_token_ids.numel()
    sens = profile_sensitivity(model, processor, calib, action_len)
    bits = allocate_bits(sens, candidate_bits, args.qvla_target_bits)
    model.zero_grad(set_to_none=True)
    def bits_for(full, lin):
        return bits.get(full, args.qvla_target_bits)  # default uniform if unseen
    return bits_for, None  # QVLA is weight-only here


def _suites(args):
    # Calibrate on all four suites unless a single --suite is being evaluated.
    return [args.suite] if getattr(args, "suite", None) else \
        ["libero_spatial", "libero_object", "libero_goal", "libero_long"]
