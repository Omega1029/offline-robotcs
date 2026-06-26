#!/usr/bin/env python3
"""
LEARNING DEMO: how quantization is actually applied — to a layer's WEIGHTS and to the
ACTIVATION values we wiretapped. Run on layer 15's mlp.down_proj of the 90.25% model.

Quantization in one line: map a real number r to a small integer q and back.
    q  = round(r / scale)            # scale = (max abs value) / (largest int)
    r̂ = q * scale                   # dequantized — close to r, but snapped to a grid
Fewer bits  ->  fewer integer levels  ->  coarser grid  ->  bigger rounding error.

We show: (1) per-channel WEIGHT quant at INT8 vs INT4, (2) per-tensor ACTIVATION quant
and why the outlier wrecks it, (3) the end-to-end output error of W4A8 vs bf16.
"""
import json
from pathlib import Path
import h5py, numpy as np, torch
from PIL import Image

import importlib.util
# This script lives in quant_expirements/; anchor all repo paths to the parent
# pipeline dir so they resolve no matter where the script is launched from.
PIPE = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("eval_libero", str(PIPE / "02_eval_libero.py"))
eval_libero = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(eval_libero)

CKPT = str(PIPE / "checkpoints/openvla_oft_libero/epoch_003/hf_model")
HEAD = str(PIPE / "checkpoints/openvla_oft_libero/epoch_003/action_head.pt")
DATA = "/home/justin_williams1/OfflineAutonomy/offline-robotcs/datasets/libero/libero_spatial"
SYS = ("A chat between a curious user and an artificial intelligence assistant. "
       "The assistant gives helpful, detailed, and polite answers to the user's questions. ")


# ── the quantizers (quant -> dequant; "fake quant" used to measure accuracy) ──
def quant_weight_per_channel(W, bits):
    """Symmetric per-OUTPUT-channel: each row gets its own scale from its own max|.|."""
    qmax = 2 ** (bits - 1) - 1                       # INT8 -> 127, INT4 -> 7
    scale = W.abs().amax(dim=1, keepdim=True) / qmax # (out, 1)
    q = torch.round(W / scale).clamp(-qmax - 1, qmax)
    return q * scale, scale                          # dequantized, scales

def quant_act_per_tensor(x, bits):
    """Symmetric per-TENSOR: ONE scale for the whole activation (the outlier sets it)."""
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().max() / qmax
    q = torch.round(x / scale).clamp(-qmax - 1, qmax)
    return q * scale, scale

def quant_act_per_token(x, bits):
    """Symmetric per-TOKEN: each sequence position gets its own scale (outlier contained)."""
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax(dim=-1, keepdim=True) / qmax
    q = torch.round(x / scale).clamp(-qmax - 1, qmax)
    return q * scale, scale

def rel_err(a, b):
    return (a - b).abs().mean().item() / (a.abs().mean().item() + 1e-9)


def one_frame(processor, model, device):
    fp = sorted(Path(DATA).glob("*.hdf5"))[0]
    with h5py.File(fp, "r") as f:
        instr = json.loads(f["data"].attrs["problem_info"])["language_instruction"].strip().rstrip(".")
        demo = f["data"]["demo_0"]; t = demo["actions"].shape[0] // 2
        img = Image.fromarray(demo["obs"]["agentview_rgb"][t][::-1, ::-1])
    prefix = f"{SYS}USER: What action should the robot take to {instr}? ASSISTANT: "
    pid = processor.tokenizer(prefix, add_special_tokens=True, return_tensors="pt")["input_ids"].squeeze(0)
    iid = torch.cat([pid, model.eval_action_token_ids]).unsqueeze(0).to(device)
    pv = processor.image_processor(images=img, return_tensors="pt")["pixel_values"].to(device, torch.bfloat16)
    return dict(input_ids=iid, attention_mask=torch.ones_like(iid), pixel_values=pv)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    processor, model = eval_libero.load_model(CKPT, HEAD, device)
    layer = model.language_model.model.layers[15].mlp.down_proj

    cap = {}
    h = layer.register_forward_hook(lambda m, i, o: cap.__setitem__("x", i[0].detach().float().cpu()))
    with torch.no_grad():
        model(**one_frame(processor, model, device))
    h.remove()

    W = layer.weight.data.detach().float().cpu()     # (4096, 11008)
    x = cap["x"]                                      # (1, 380, 11008)

    print("\n================ (1) WEIGHT QUANTIZATION ================")
    print(f"W shape {tuple(W.shape)}  range [{W.min():.3f}, {W.max():.3f}]")
    for b in (8, 4):
        Wdq, scale = quant_weight_per_channel(W, b)
        print(f"  INT{b}: {2**b:>3d} levels | example row scale={scale[0].item():.5f} | "
              f"relative weight error = {rel_err(Wdq, W)*100:.2f}%")

    print("\n================ (2) ACTIVATION QUANTIZATION ============")
    print(f"x range [{x.min():.3f}, {x.max():.3f}]  (outlier |max| = {x.abs().max():.3f})")
    xq_t, s_t = quant_act_per_tensor(x, 8)
    typ = x.std().item()
    print(f"  INT8 per-TENSOR : scale={s_t:.4f} (step size) | rel err = {rel_err(xq_t, x)*100:.2f}%")
    print(f"     a typical value (~1 std = {typ:.3f}) only spans ~{typ/s_t:.1f} integer levels "
          f"of the 255 available — the outlier ate the range.")
    xq_k, _ = quant_act_per_token(x, 8)
    print(f"  INT8 per-TOKEN  : rel err = {rel_err(xq_k, x)*100:.2f}%  "
          f"(finer-grained scaling contains the outlier)")

    print("\n================ (3) END-TO-END LAYER OUTPUT ============")
    y = x @ W.T                                       # bf16-equivalent reference output
    Wdq8, _ = quant_weight_per_channel(W, 8); Wdq4, _ = quant_weight_per_channel(W, 4)
    for tag, Wq, xq in [("W8A8 (per-token act)", Wdq8, quant_act_per_token(x, 8)[0]),
                        ("W4A8 (per-token act)", Wdq4, quant_act_per_token(x, 8)[0]),
                        ("W4A16 (act in fp)",    Wdq4, x)]:
        yq = xq @ Wq.T
        print(f"  {tag:22s}: output relative error = {rel_err(yq, y)*100:.2f}%")
    print("=========================================================")
    print("Takeaway: INT8 weights ~lossless; INT4 weights add error; the activation "
          "outlier makes per-tensor act-quant far worse than per-token. This output error "
          "is what compounds over 32 layers into the success-rate drop the eval measures.")


if __name__ == "__main__":
    main()
