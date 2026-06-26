#!/usr/bin/env python3
"""
EXPERIMENT 08B: QuantVLA (arXiv 2602.20309, "Scale-Calibrated PTQ for VLA") applied to
ONE layer — layer 15's mlp.down_proj of the 90.25% OpenVLA-OFT model — so we can see
its three components in isolation and compare against naive max-quant (08_quant_demo.py).

QuantVLA is training-free PTQ using a small unlabeled CALIBRATION BUFFER. Its abstract
describes three parts; here is how each maps onto this MLP layer:

  (A) Selective quantization layout
        "integerizes all linear layers in the language backbone and the DiT while keeping
         attention projections in floating point."
      => down_proj is an MLP linear, so it IS quantized (W + A). Attention q/k/v/o would
         stay FP. (We're on an MLP layer, so the layout rule says: quantize this one.)

  (B) Scale-calibrated quantization  (the "scale-calibrated" in the title)
        Instead of naive min/max scales, pick scales from the calibration buffer that
        minimize quantization MSE — per-output-channel for weights, a static calibrated
        per-tensor scale for activations (defuses the outlier we saw in 08).

  (C) Output head balancing
        "per-layer residual interface calibration that mitigates post-projection energy
         drift." => after quantizing, the layer output has a small systematic bias vs fp;
         we estimate per-output-channel bias on the calib buffer and fold it back in
         (classic bias-correction).

  Attention temperature matching is QuantVLA's 4th idea but it is ATTENTION-specific
  (rescales attention logits); it does not apply to this MLP layer, so it's a documented
  no-op here (see attention_temperature_matching()).

NOTE: the paper's abstract does not state exact bit-widths; W4 weights + INT8 activations
are used as a reasonable "scale-calibrated PTQ" default and are CLI-configurable. These
choices are our interpretation (no official code released).

Run it yourself, e.g.:
  CUDA_VISIBLE_DEVICES=3 MUJOCO_GL=egl python quant_expirements/08B_quantvla_layer.py
"""
import argparse, json
from pathlib import Path
import h5py, numpy as np, torch
from PIL import Image

import importlib.util
# Same anchoring as 08_quant_demo.py: paths resolve from the parent pipeline dir.
PIPE = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("eval_libero", str(PIPE / "02_eval_libero.py"))
eval_libero = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(eval_libero)

CKPT = str(PIPE / "checkpoints/openvla_oft_libero/epoch_003/hf_model")
HEAD = str(PIPE / "checkpoints/openvla_oft_libero/epoch_003/action_head.pt")
DATA = "/home/justin_williams1/OfflineAutonomy/offline-robotcs/datasets/libero/libero_spatial"
SYS = ("A chat between a curious user and an artificial intelligence assistant. "
       "The assistant gives helpful, detailed, and polite answers to the user's questions. ")


# ───────────────────────────── calibration buffer ────────────────────────────
def build_inputs(processor, model, device, instr, img):
    prefix = f"{SYS}USER: What action should the robot take to {instr}? ASSISTANT: "
    pid = processor.tokenizer(prefix, add_special_tokens=True, return_tensors="pt")["input_ids"].squeeze(0)
    iid = torch.cat([pid, model.eval_action_token_ids]).unsqueeze(0).to(device)
    pv = processor.image_processor(images=img, return_tensors="pt")["pixel_values"].to(device, torch.bfloat16)
    return dict(input_ids=iid, attention_mask=torch.ones_like(iid), pixel_values=pv)


def collect_frames(n):
    """Yield (instruction, PIL image) pairs spread across libero_spatial demos."""
    out = []
    for fp in sorted(Path(DATA).glob("*.hdf5")):
        with h5py.File(fp, "r") as f:
            instr = json.loads(f["data"].attrs["problem_info"])["language_instruction"].strip().rstrip(".")
            demo = f["data"]["demo_0"]; T = demo["actions"].shape[0]
            for t in np.linspace(0, T - 1, 3, dtype=int):
                out.append((instr, Image.fromarray(demo["obs"]["agentview_rgb"][t][::-1, ::-1])))
                if len(out) >= n:
                    return out
    return out


def capture_layer_io(model, processor, layer, device, frames):
    """Run frames through the model; capture this layer's INPUT activation and fp OUTPUT.
    Returns X (Ntok, in), Yfp (Ntok, out) concatenated over all calibration tokens."""
    xs, ys = [], []
    def hook(m, i, o):
        xs.append(i[0].detach().float().cpu().reshape(-1, i[0].shape[-1]))
        ys.append(o.detach().float().cpu().reshape(-1, o.shape[-1]))
    h = layer.register_forward_hook(hook)
    with torch.no_grad():
        for instr, img in frames:
            model(**build_inputs(processor, model, device, instr, img))
    h.remove()
    return torch.cat(xs, 0), torch.cat(ys, 0)


# ─────────────────── (B) scale-calibrated quantizers ─────────────────────────
def calibrated_weight_quant(W, bits, ratios=(1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6)):
    """Per-output-channel symmetric weight quant; each row's clip ratio is chosen to
    MINIMIZE that row's quant MSE (scale calibration), not just raw max."""
    qmax = 2 ** (bits - 1) - 1
    amax = W.abs().amax(dim=1, keepdim=True)                  # (out,1)
    best_mse = torch.full((W.shape[0], 1), float("inf"))
    best_scale = amax / qmax
    for r in ratios:
        scale = (amax * r).clamp_min(1e-8) / qmax
        Wq = torch.round(W / scale).clamp(-qmax - 1, qmax) * scale
        mse = ((Wq - W) ** 2).mean(dim=1, keepdim=True)
        better = mse < best_mse
        best_mse = torch.where(better, mse, best_mse)
        best_scale = torch.where(better, scale, best_scale)
    Wq = torch.round(W / best_scale).clamp(-qmax - 1, qmax) * best_scale
    return Wq, best_scale


def calibrated_act_scale(X, bits, ratios=(1.0, 0.99, 0.98, 0.95, 0.9, 0.8)):
    """Static per-TENSOR activation scale chosen on the calib buffer to minimize MSE
    (a fixed scale folded into the layer at inference — what 'integerizes activations'
    with calibration means). Clipping the outlier trades a little clip error for much
    finer resolution on the bulk."""
    qmax = 2 ** (bits - 1) - 1
    amax = X.abs().max()
    best = (float("inf"), amax / qmax)
    for r in ratios:
        scale = (amax * r) / qmax
        Xq = torch.round((X / scale).clamp(-qmax - 1, qmax)) * scale
        mse = ((Xq - X) ** 2).mean().item()
        if mse < best[0]:
            best = (mse, scale)
    return best[1]


def quant_act(X, scale, bits):
    qmax = 2 ** (bits - 1) - 1
    return torch.round((X / scale).clamp(-qmax - 1, qmax)) * scale


# ─────────────────── (C) output head balancing (bias correction) ─────────────
def output_balancing(Yfp, Yq):
    """Per-output-channel bias = mean(fp - quant) over the calib buffer; added at
    inference to cancel the systematic 'energy drift' the paper describes."""
    return (Yfp - Yq).mean(dim=0)                            # (out,)


# ─────────────────── attention temperature matching (N/A here) ───────────────
def attention_temperature_matching(*_):
    """QuantVLA's per-head logit rescaling — ATTENTION-only. down_proj is an MLP layer,
    so this is a documented no-op. It would apply to self_attn.{q,k}_proj when we extend
    08B to an attention layer."""
    return None


def rel(a, b):
    return (a - b).abs().mean().item() / (b.abs().mean().item() + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w_bits", type=int, default=4)
    ap.add_argument("--a_bits", type=int, default=8)
    ap.add_argument("--calib_frames", type=int, default=24)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    processor, model = eval_libero.load_model(CKPT, HEAD, device)
    layer = model.language_model.model.layers[15].mlp.down_proj
    W = layer.weight.data.detach().float().cpu()             # (4096, 11008)

    frames = collect_frames(args.calib_frames)
    print(f"[08B] calib buffer: {len(frames)} frames; layer = layers[15].mlp.down_proj "
          f"(W{args.w_bits}A{args.a_bits})")
    X, Yfp = capture_layer_io(model, processor, layer, device, frames)
    # hold out the last frame's tokens as the test split
    n_test = max(1, X.shape[0] // (len(frames)))             # ~one frame of tokens
    Xtr, Ytr = X[:-n_test], Yfp[:-n_test]
    Xte, Yte = X[-n_test:], Yfp[-n_test:]
    print(f"[08B] calib tokens={Xtr.shape[0]}  test tokens={Xte.shape[0]}")

    # (A) selective layout: this MLP linear is quantized.
    # (B) scale-calibrated weight + activation quant, calibrated on the buffer.
    Wq, _ = calibrated_weight_quant(W, args.w_bits)
    a_scale = calibrated_act_scale(Xtr, args.a_bits)
    attention_temperature_matching(layer)                    # no-op (MLP)

    # (C) output balancing: estimate bias on calib, apply on test.
    Ytr_q = quant_act(Xtr, a_scale, args.a_bits) @ Wq.T
    bias = output_balancing(Ytr, Ytr_q)

    # ── evaluate on held-out test tokens ─────────────────────────────────────
    Yte_quantvla = quant_act(Xte, a_scale, args.a_bits) @ Wq.T + bias

    # baseline: naive 08-style per-tensor max activation + raw-max weight quant
    qmaxw = 2 ** (args.w_bits - 1) - 1
    sW = W.abs().amax(dim=1, keepdim=True) / qmaxw
    Wq_naive = torch.round(W / sW).clamp(-qmaxw - 1, qmaxw) * sW
    sA = Xte.abs().max() / (2 ** (args.a_bits - 1) - 1)
    Yte_naive = quant_act(Xte, sA, args.a_bits) @ Wq_naive.T

    print("\n================ QuantVLA vs naive (held-out output error) ================")
    print(f"  naive  max-quant            : {rel(Yte_naive, Yte)*100:.2f}%")
    print(f"  + scale-calibrated W&A (B)  : {rel(quant_act(Xte, a_scale, args.a_bits) @ Wq.T, Yte)*100:.2f}%")
    print(f"  + output balancing (C)      : {rel(Yte_quantvla, Yte)*100:.2f}%   <- full QuantVLA path")
    mem = 100 * (1 - args.w_bits / 16)
    print(f"\n  weight memory savings @ W{args.w_bits}: ~{mem:.0f}% vs bf16 (this layer)")
    print("  (A) layout: MLP linear quantized, attention proj would stay FP")
    print("  (D) attention temperature matching: N/A on MLP (no-op)")
    print("===========================================================================")


if __name__ == "__main__":
    main()
