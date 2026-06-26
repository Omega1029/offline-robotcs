#!/usr/bin/env python3
"""
LEARNING DEMO: a PyTorch forward-hook "wiretap" on the trained 90.25% OpenVLA-OFT model.

We attach ONE hook to ONE layer, push ONE real LIBERO frame through the model, and
print exactly what the hook sees. Nothing is quantized here — this is purely to watch
the mechanism work end-to-end.

What a forward hook is:
  module.register_forward_hook(fn) registers fn(module, input, output). PyTorch calls
  fn SYNCHRONOUSLY, right after `module.forward()` runs, every time data flows through
  that module. `input` is a TUPLE of the positional args the module received; `output`
  is whatever it returned. For an nn.Linear, input[0] is the activation going IN
  (shape [batch, seq, in_features]) and output is input[0] @ W.T + b
  (shape [batch, seq, out_features]).

  The hook does NOT move anything off-GPU by itself — we choose to .detach().cpu() the
  tensor inside the hook to "save a copy at that microsecond." We .remove() the handle
  afterward so it doesn't fire on every future forward (a common memory/perf leak).

Run:
  CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl python 07_hook_demo.py
"""
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

# Reuse the trained-model loader + prompt conventions from the fp eval.
import importlib.util
_spec = importlib.util.spec_from_file_location("eval_libero", str(Path(__file__).parent / "02_eval_libero.py"))
eval_libero = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(eval_libero)

CKPT = "checkpoints/openvla_oft_libero/epoch_003/hf_model"
HEAD = "checkpoints/openvla_oft_libero/epoch_003/action_head.pt"
DATA = "/home/justin_williams1/OfflineAutonomy/offline-robotcs/datasets/libero/libero_spatial"
SYS_MSG = ("A chat between a curious user and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the user's questions. ")


def one_libero_frame(processor, model, device):
    """Build a single forward input (prompt + action tokens + one image) — the same
    recipe 02_eval_libero.run_rollout uses, so the backbone is exercised exactly as in eval."""
    fp = sorted(Path(DATA).glob("*.hdf5"))[0]
    with h5py.File(fp, "r") as f:
        instr = json.loads(f["data"].attrs["problem_info"])["language_instruction"].strip().rstrip(".")
        demo = f["data"]["demo_0"]
        t = demo["actions"].shape[0] // 2                       # a frame from the middle
        img = Image.fromarray(demo["obs"]["agentview_rgb"][t][::-1, ::-1])  # 180° flip (matches training)
    prefix = f"{SYS_MSG}USER: What action should the robot take to {instr}? ASSISTANT: "
    prefix_ids = processor.tokenizer(prefix, add_special_tokens=True, return_tensors="pt")["input_ids"].squeeze(0)
    input_ids = torch.cat([prefix_ids, model.eval_action_token_ids]).unsqueeze(0).to(device)
    pixel_values = processor.image_processor(images=img, return_tensors="pt")["pixel_values"].to(device, torch.bfloat16)
    print(f"[Demo] instruction: {instr!r}")
    return dict(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), pixel_values=pixel_values)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    processor, model = eval_libero.load_model(CKPT, HEAD, device)  # the 90.25% model
    model.eval()

    # ── pick ONE layer to wiretap ────────────────────────────────────────────
    # layer 15's MLP down-projection: input is the 11008-wide hidden, output is 4096-wide.
    target_name = "language_model.model.layers[15].mlp.down_proj"
    target = model.language_model.model.layers[15].mlp.down_proj
    print(f"[Hook] target layer: {target_name}  ({target})")

    captured = {}

    def wiretap(module, inputs, output):
        # inputs is a TUPLE; inputs[0] is the activation flowing INTO this Linear.
        x_in = inputs[0].detach().float().cpu()      # copy off-GPU "at this microsecond"
        y_out = output.detach().float().cpu()
        captured["in"] = x_in
        captured["out"] = y_out

    handle = target.register_forward_hook(wiretap)   # attach the wiretap

    batch = one_libero_frame(processor, model, device)
    with torch.no_grad():
        model(**batch, output_hidden_states=False)   # one forward → hook fires once

    handle.remove()                                  # ALWAYS detach the wiretap

    # ── inspect what we captured ─────────────────────────────────────────────
    x, y = captured["in"], captured["out"]
    print("\n================ WHAT THE HOOK CAPTURED ================")
    print(f"INPUT  activation: shape={tuple(x.shape)}  dtype={x.dtype}  "
          f"(batch, sequence_length, in_features=11008)")
    print(f"OUTPUT activation: shape={tuple(y.shape)}  (… out_features=4096)")
    print(f"\nINPUT  stats : min={x.min():.3f}  max={x.max():.3f}  mean={x.mean():.4f}  std={x.std():.3f}")
    print(f"OUTPUT stats : min={y.min():.3f}  max={y.max():.3f}  mean={y.mean():.4f}  std={y.std():.3f}")
    print(f"\nFirst 8 values of input[0,0,:8]:\n  {x[0,0,:8].numpy().round(4)}")

    # The quant-relevant view: per-INPUT-CHANNEL max magnitude over the sequence.
    # A few channels with huge values = the 'outlier features' that make low-bit hard.
    per_channel_max = x[0].abs().amax(dim=0)          # (11008,)
    top = torch.topk(per_channel_max, 5)
    print(f"\nPer-input-channel |max| — median={per_channel_max.median():.3f}, "
          f"overall max={per_channel_max.max():.3f}")
    print(f"  Top-5 outlier channels: idx={top.indices.tolist()}  vals={top.values.numpy().round(2).tolist()}")
    print(f"  (outlier/median ratio ≈ {per_channel_max.max()/per_channel_max.median():.1f}× — "
          f"this spread is exactly what activation-aware quant must handle.)")
    print("========================================================")


if __name__ == "__main__":
    main()
