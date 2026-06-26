#!/usr/bin/env python3
"""
quant.py — reusable quantization techniques for experimenting on OpenVLA-OFT.

All techniques operate on the Llama BACKBONE linears (vision tower / projector / lm_head /
action head stay bf16 — that skip list is in `backbone_linears`). Everything is
SIMULATED (fake) quantization: quantize→dequantize in floating point, so you can measure
accuracy without low-bit kernels.

Composition model
-----------------
`FakeQuant.Linear` is the universal drop-in module: give it a dequantized weight tensor
(and optional activation scale / bias correction) and it behaves like the quantized layer.
Every technique just decides *what* weight_dq / a_scale / bias to hand it:

    FakeQuant   — uniform per-channel weight quant (+ optional activation quant). The baseline.
    QVLA        — per-channel, action-sensitivity bit allocation (incl. 0-bit pruning).
    QuantVLA    — scale-calibrated PTQ: MSE-calibrated scales + output (bias) balancing.

Each technique exposes:
    .name
    .apply(model, ...)   -> grafts FakeQuant.Linear modules into the backbone (in place)

Add a new method by subclassing nothing in particular — just produce weight_dq / a_scale /
bias for each layer and wrap with FakeQuant.Linear.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

SKIP_SUBSTRINGS = ("vision_backbone", "projector", "lm_head")


def backbone_linears(model):
    """Yield (full_name, parent_module, attr, nn.Linear) for every quantizable backbone
    linear (everything under language_model except the skip list)."""
    lm = model.language_model
    for mod_name, module in lm.named_modules():
        for attr, child in list(module.__dict__.get("_modules", {}).items()):
            if isinstance(child, nn.Linear):
                full = f"language_model.{mod_name}.{attr}".replace("..", ".")
                if any(s in full for s in SKIP_SUBSTRINGS):
                    continue
                yield full, module, attr, child


def _module_linears(root, prefix):
    """Yield (full_name, parent_module, attr, nn.Linear) for every nn.Linear under `root`."""
    if root is None:
        return
    for mod_name, module in root.named_modules():
        for attr, child in list(module.__dict__.get("_modules", {}).items()):
            if isinstance(child, nn.Linear):
                full = f"{prefix}.{mod_name}.{attr}".replace("..", ".")
                yield full, module, attr, child


def vision_linears(model):
    """Yield quantizable nn.Linear in the fused vision tower (DINOv2 + SigLIP featurizers).
    These are the ViT attention qkv/proj and MLP fc1/fc2 layers."""
    yield from _module_linears(getattr(model, "vision_backbone", None), "vision_backbone")


def action_head_linears(model):
    """Yield nn.Linear in the L1RegressionActionHead (MLPResNet: fc1, block ffns, fc2).
    The most action-sensitive component — quantize with care (channel-wise)."""
    yield from _module_linears(getattr(model, "action_head", None), "action_head")


# ════════════════════════════════ FakeQuant ══════════════════════════════════
class FakeQuant:
    """Core primitives (static) + the universal quantized-Linear wrapper + a uniform
    baseline `.apply`."""
    name = "fakequant"

    def __init__(self, w_bits: int = 8, a_bits: int | None = None, a_mode: str = "per_token"):
        self.w_bits, self.a_bits, self.a_mode = w_bits, a_bits, a_mode

    # ---- primitives -----------------------------------------------------------
    @staticmethod
    def quant_weight_per_channel(W: torch.Tensor, bits, scale: torch.Tensor | None = None):
        """Symmetric per-output-channel weight quant. `bits` may be an int or a
        per-row (out_features,) int tensor; bits==0 prunes that row to zero."""
        W = W.float()
        if isinstance(bits, int):
            bits = torch.full((W.shape[0],), bits, dtype=torch.int32)
        out = torch.empty_like(W)
        for b in bits.unique().tolist():
            rows = bits == b
            if b == 0:
                out[rows] = 0.0
                continue
            wr = W[rows]
            qmax = 2 ** (b - 1) - 1
            s = wr.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax if scale is None else scale[rows]
            out[rows] = torch.round(wr / s).clamp(-qmax - 1, qmax) * s
        return out

    @staticmethod
    def quant_weight_fp8(W: torch.Tensor, fmt: str = "e4m3", per_channel: bool = True):
        """Fake FP8 weight quant using torch's real FP8 dtypes (true E4M3/E5M2 rounding),
        with a per-output-channel (or per-tensor) scale into the FP8 dynamic range."""
        W = W.float()
        if fmt == "e4m3":
            dt, fmax = torch.float8_e4m3fn, 448.0
        else:  # e5m2
            dt, fmax = torch.float8_e5m2, 57344.0
        if per_channel:
            s = W.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / fmax
        else:
            s = W.abs().max().clamp_min(1e-8) / fmax
        return (W / s).clamp(-fmax, fmax).to(dt).float() * s

    @staticmethod
    def quant_act(x: torch.Tensor, bits: int, scale: torch.Tensor | float | None = None,
                  mode: str = "per_token"):
        """Symmetric activation quant. mode: 'per_tensor' (one scale; static if `scale`
        given) or 'per_token' (dynamic per sequence position)."""
        if bits is None or bits >= 16:
            return x
        qmax = 2 ** (bits - 1) - 1
        if scale is None:
            if mode == "per_token":
                scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
            else:
                scale = x.abs().max().clamp_min(1e-8) / qmax
        return torch.round((x / scale).clamp(-qmax - 1, qmax)) * scale

    @staticmethod
    def mse_scale(t: torch.Tensor, bits: int, dim=None,
                  ratios=(1.0, 0.99, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6)):
        """Clip-ratio scale that minimizes quant MSE. dim=None → scalar (per-tensor);
        dim=1 → per-row (per output channel)."""
        qmax = 2 ** (bits - 1) - 1
        amax = t.abs().amax(dim=dim, keepdim=True) if dim is not None else t.abs().max()
        best_mse = torch.full_like(amax, float("inf")) if dim is not None else torch.tensor(float("inf"))
        best = amax / qmax
        for r in ratios:
            s = (amax * r).clamp_min(1e-8) / qmax
            tq = torch.round((t / s).clamp(-qmax - 1, qmax)) * s
            mse = ((tq - t) ** 2).mean(dim=dim, keepdim=True) if dim is not None else ((tq - t) ** 2).mean()
            better = mse < best_mse
            best_mse = torch.where(better, mse, best_mse)
            best = torch.where(better, s, best)
        return best

    # ---- the universal quantized layer ---------------------------------------
    class Linear(nn.Module):
        """Drop-in for nn.Linear. Supply a dequantized weight (or let it self-quantize at
        w_bits), plus optional activation quant (a_bits/a_scale/a_mode) and a per-output-
        channel bias correction. Any technique composes by handing these in."""
        def __init__(self, linear: nn.Linear, weight_dq: torch.Tensor | None = None,
                     w_bits: int = 8, a_bits: int | None = None, a_scale=None,
                     a_mode: str = "per_token", bias_correction: torch.Tensor | None = None):
            super().__init__()
            if weight_dq is None:
                weight_dq = FakeQuant.quant_weight_per_channel(linear.weight.data, w_bits)
            self.register_buffer("weight", weight_dq.to(linear.weight.dtype))
            self.bias = linear.bias
            self.a_bits, self.a_mode = a_bits, a_mode
            if a_scale is not None and not torch.is_tensor(a_scale):
                a_scale = torch.tensor(a_scale)
            self.register_buffer("a_scale", a_scale if a_scale is not None else None)
            self.register_buffer("bias_corr",
                                 bias_correction.to(linear.weight.dtype) if bias_correction is not None else None)

        def forward(self, x):
            if self.a_bits is not None:
                x = FakeQuant.quant_act(x.float(), self.a_bits, scale=self.a_scale,
                                        mode=self.a_mode).to(self.weight.dtype)
            y = F.linear(x, self.weight, self.bias)
            if self.bias_corr is not None:
                y = y + self.bias_corr
            return y

    # ---- model-level apply (uniform baseline) --------------------------------
    def apply(self, model):
        n = 0
        for full, parent, attr, lin in list(backbone_linears(model)):
            setattr(parent, attr, FakeQuant.Linear(lin, w_bits=self.w_bits,
                                                    a_bits=self.a_bits, a_mode=self.a_mode
                                                    ).to(lin.weight.device))
            n += 1
        print(f"[FakeQuant] grafted {n} backbone linears @ W{self.w_bits}"
              f"{'A'+str(self.a_bits) if self.a_bits else ''} ({self.a_mode})")
        return model


# ═══════════════════════════════ ComponentQuant ══════════════════════════════
class ComponentQuant:
    """Phase-1 component-wise quantization: independently set precision for the vision
    tower, the Llama backbone, and the action head, then graft FakeQuant.Linear into each.

    Spec per component is one of:
        None          -> leave bf16 (untouched)
        int <bits>    -> symmetric per-output-channel weight quant at <bits>
        "fp8" / "fp8_e5m2" -> real FP8 (E4M3 / E5M2) weight quant
    `backbone_a_bits` optionally adds activation quant on the backbone (e.g. W4A4).
    The backbone may also be handed off to an external technique via `backbone_tech`
    (e.g. a DCT-rotation Transform), in which case its bits/a_bits here are ignored.
    """
    name = "component"

    def __init__(self, vision=None, backbone=None, head=None,
                 backbone_a_bits=None, head_mode="per_channel", backbone_tech=None):
        self.vision = vision
        self.backbone = backbone
        self.head = head
        self.backbone_a_bits = backbone_a_bits
        self.head_mode = head_mode
        self.backbone_tech = backbone_tech

    @staticmethod
    def _graft(enum, spec, device, a_bits=None):
        """Graft FakeQuant.Linear into every linear yielded by `enum` per `spec`."""
        if spec is None:
            return 0
        n = 0
        for full, parent, attr, lin in list(enum):
            if isinstance(spec, str) and spec.startswith("fp8"):
                fmt = "e5m2" if "e5m2" in spec else "e4m3"
                wdq = FakeQuant.quant_weight_fp8(lin.weight.data, fmt=fmt)
                ql = FakeQuant.Linear(lin, weight_dq=wdq, a_bits=a_bits)
            else:  # integer bits
                ql = FakeQuant.Linear(lin, w_bits=int(spec), a_bits=a_bits)
            setattr(parent, attr, ql.to(device))
            n += 1
        return n

    def apply(self, model, ctx=None):
        dev = next(model.parameters()).device
        # Backbone: either an external technique (e.g. DCT rotation) or plain int/fp8.
        if self.backbone_tech is not None:
            import inspect
            p = inspect.signature(self.backbone_tech.apply).parameters
            (self.backbone_tech.apply(model, ctx) if len(p) >= 2
             else self.backbone_tech.apply(model))
            nb = "<tech>"
        else:
            nb = self._graft(backbone_linears(model), self.backbone, dev,
                             a_bits=self.backbone_a_bits)
        nv = self._graft(vision_linears(model), self.vision, dev)
        nh = self._graft(action_head_linears(model), self.head, dev)
        print(f"[ComponentQuant] vision={self.vision}({nv})  "
              f"backbone={self.backbone if self.backbone_tech is None else 'tech'}({nb})  "
              f"head={self.head}({nh})")
        # effective backbone bits for size accounting
        if self.backbone_tech is not None:
            return float(getattr(self.backbone_tech, "w_bits", 4) or 4)
        return float(self.backbone or 16)


# ═══════════════════════════════════ QVLA ════════════════════════════════════
class QVLA:
    """Action-centric, per-channel bit allocation (arXiv 2602.03782).

    Pipeline:  sens = profile(model, calib_inputs);  apply(model, sens)
      profile : one fwd+bwd per calib sample → per-output-channel action-gradient g_c
                and mean input scale (the action-space sensitivity surrogate).
      apply   : Lagrangian per-channel bit allocation over candidate_bits (0-bit = prune)
                to the target average bit-width, then graft FakeQuant.Linear with those bits.
    """
    name = "qvla"

    def __init__(self, target_bits: float = 4.0, candidate_bits=(0, 2, 3, 4, 8)):
        self.target_bits = target_bits
        self.candidate_bits = list(candidate_bits)

    @torch.enable_grad()
    def profile(self, model, calib_inputs):
        """calib_inputs: list of forward-input dicts (input_ids/attention_mask/pixel_values).
        Requires model.eval_action_token_ids + model.action_head (set by load_model)."""
        action_len = model.eval_action_token_ids.numel()
        layers = {n: lin for n, _, _, lin in backbone_linears(model)}
        in_scale = {n: 0.0 for n in layers}
        g_acc = {n: torch.zeros(l.out_features, device=l.weight.device) for n, l in layers.items()}

        handles, outs = [], {}
        def mk(name):
            def hook(m, inp, out):
                in_scale[name] += inp[0].detach().abs().mean().item()
                out.requires_grad_(True); out.retain_grad(); outs[name] = out
            return hook
        for n, l in layers.items():
            handles.append(l.register_forward_hook(mk(n)))

        used = 0
        for b in calib_inputs:
            outs.clear()
            o = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                      pixel_values=b["pixel_values"], output_hidden_states=True)
            action = model.action_head.predict_action(o.hidden_states[-1][:, -action_len:, :])
            model.zero_grad(set_to_none=True)
            action.float().abs().sum().backward()
            for n in layers:
                g = outs.get(n)
                if g is not None and g.grad is not None:
                    g_acc[n] += g.grad.detach().abs().reshape(-1, g.shape[-1]).mean(0)
            used += 1
        for h in handles:
            h.remove()
        model.zero_grad(set_to_none=True)
        sens = {n: {"g": (g_acc[n] / max(used, 1)).float().cpu(),
                    "in_scale": in_scale[n] / max(used, 1), "lin": l}
                for n, l in layers.items()}
        print(f"[QVLA] profiled {len(sens)} layers over {used} samples")
        return sens

    def _alloc(self, sens):
        names, costs, spans, cur = [], [], [], 0
        for n, d in sens.items():
            W = d["lin"].weight.data
            g = d["g"].to(W.device)
            # quant ERROR per channel at each candidate bit (b=0 → full weight norm = pruning error)
            cols = []
            for b in self.candidate_bits:
                if b == 0:
                    err = W.float().norm(dim=1)
                else:
                    qmax = 2 ** (b - 1) - 1
                    s = W.float().abs().amax(1, keepdim=True).clamp_min(1e-8) / qmax
                    Wq = torch.round(W.float() / s).clamp(-qmax - 1, qmax) * s
                    err = (W.float() - Wq).norm(dim=1)
                cols.append(g * err * d["in_scale"])
            costs.append(torch.stack(cols, 1).float().cpu())
            names.append(n); spans.append((cur, cur + W.shape[0])); cur += W.shape[0]
        cost = torch.cat(costs, 0)
        cand = torch.tensor(self.candidate_bits, dtype=torch.float32)
        lo, hi = 0.0, 1e6
        for _ in range(50):
            mid = (lo + hi) / 2
            avg = cand[(cost + mid * cand).argmin(1)].mean().item()
            lo, hi = (mid, hi) if avg > self.target_bits else (lo, mid)
        idx = (cost + hi * cand).argmin(1)
        bits = {n: torch.tensor([self.candidate_bits[i] for i in idx[s:e].tolist()], dtype=torch.int32)
                for n, (s, e) in zip(names, spans)}
        from collections import Counter
        print(f"[QVLA] avg bits={cand[idx].mean():.2f} (target {self.target_bits}) "
              f"dist={dict(sorted(Counter(self.candidate_bits[i] for i in idx.tolist()).items()))}")
        return bits

    def apply(self, model, sens):
        bits = self._alloc(sens)
        for full, parent, attr, lin in list(backbone_linears(model)):
            wq = FakeQuant.quant_weight_per_channel(lin.weight.data, bits[full].to(lin.weight.device))
            setattr(parent, attr, FakeQuant.Linear(lin, weight_dq=wq).to(lin.weight.device))
        return model


# ═════════════════════════════════ QuantVLA ══════════════════════════════════
class QuantVLA:
    """Scale-calibrated PTQ (arXiv 2602.20309).

    Per layer:  (B) per-output-channel MSE-calibrated weight scale + static MSE-calibrated
    per-tensor activation scale;  (C) per-output-channel output bias correction estimated
    on the calibration buffer. (Attention-temperature-matching is attention-only — N/A to
    MLP linears, see note.)  Bit-widths default W4A8 (the paper's abstract omits exact bits).
    """
    name = "quantvla"

    def __init__(self, w_bits: int = 4, a_bits: int = 8):
        self.w_bits, self.a_bits = w_bits, a_bits

    def calibrate_weight(self, W):
        s = FakeQuant.mse_scale(W.float(), self.w_bits, dim=1)
        return FakeQuant.quant_weight_per_channel(W, self.w_bits, scale=s)

    def calibrate_act_scale(self, X):
        return FakeQuant.mse_scale(X.float(), self.a_bits, dim=None)

    @staticmethod
    def output_balancing(Yfp, Yq):
        return (Yfp - Yq).mean(dim=0)

    def quantize_layer(self, linear, X_calib):
        """Return a FakeQuant.Linear for this layer calibrated on X_calib (calib tokens,
        in_features). Includes weight calib, static act scale, and output bias correction."""
        W = linear.weight.data
        wq = self.calibrate_weight(W)
        a_scale = self.calibrate_act_scale(X_calib)
        Yfp = X_calib @ W.float().T
        Yq = FakeQuant.quant_act(X_calib, self.a_bits, scale=a_scale, mode="per_tensor") @ wq.float().T
        bias = self.output_balancing(Yfp, Yq)
        return FakeQuant.Linear(linear, weight_dq=wq, a_bits=self.a_bits, a_scale=a_scale,
                                a_mode="per_tensor", bias_correction=bias)
