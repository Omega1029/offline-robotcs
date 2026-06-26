#!/usr/bin/env python3
"""
quant_advanced.py — advanced (linear-algebra / statistical) quantization techniques for
OpenVLA-OFT, built on the primitives in quant.py. Everything is SIMULATED (fake) quant:
quantize→dequantize in fp so we can measure closed-loop LIBERO success without low-bit
kernels. All techniques operate ONLY on the Llama backbone linears (vision tower /
projector / lm_head / action head stay bf16 — skip list lives in quant.backbone_linears).

The frontier these target: naive per-channel W4 is already ~lossless on this model
(90.75% vs 90.25% bf16). The interesting regime is W3/W2 weights and low-bit activations
(A8/A4), where naive quant collapses and these methods aim to recover it.

Techniques (each exposes .name and .apply(model, ctx) -> effective_bits):

  Rotation     — orthogonal incoherence processing (QuaRot/SpinQuant family). Insert an
                 orthogonal Q on the input axis: Wx = (W Qᵀ)(Q x). Q mixes channels so
                 weights AND activations become Gaussian-ish, smashing the outlier channels
                 that wreck low-bit quant. Reconstructs exactly in fp (QᵀQ = I). Hadamard
                 where the dim is a power of two (FWHT-equivalent, value-bounded), random
                 orthogonal otherwise.
  AWQScale     — activation-aware weight scaling (AWQ). Per-input-channel scale s_j folds
                 salience out of the weights: W'_:,j = W_:,j·s_j, x'_j = x_j/s_j, then
                 quantize W'. α chosen by grid search to minimize per-layer output MSE on
                 the calibration buffer. Bit-budget unchanged (scales fold into layernorm).
  SVDResidual  — low-rank residual correction. Quantize W to b bits, then keep the top-r
                 singular directions of the residual R = W − Q(W) in fp16. Pays r·(out+in)
                 fp16 params for a big MSE cut. effective_bits accounts for that overhead.
  MixedPrec    — per-layer W4/W8 by sensitivity (activation-scaled weight-quant error),
                 allocated to a target average bit-width. Cheap structured mixed precision.

Compose by passing the same model through more than one (e.g. Rotation then quantize, or
AWQScale then SVDResidual) — see 09_quant_sweep.py for the registry.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import quant  # FakeQuant primitives + backbone_linears + skip list (same dir)


# ════════════════════════════ rotation utilities ═════════════════════════════
def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def hadamard_matrix(n: int, device, dtype=torch.float32) -> torch.Tensor:
    """Normalized Sylvester–Hadamard matrix (n must be a power of two). Orthogonal,
    entries ±1/√n, so it is value-bounded — the property that makes Hadamard the
    preferred incoherence rotation (no new outliers introduced)."""
    assert _is_pow2(n), f"hadamard needs power-of-two dim, got {n}"
    H = torch.ones(1, 1, device=device, dtype=dtype)
    base = torch.tensor([[1.0, 1.0], [1.0, -1.0]], device=device, dtype=dtype)
    k = int(round(math.log2(n)))
    for _ in range(k):
        H = torch.kron(H, base)
    return H / math.sqrt(n)


def random_orthogonal(n: int, device, dtype=torch.float32, seed: int = 0) -> torch.Tensor:
    """Haar-ish random orthogonal matrix via QR of a Gaussian. Used for non-power-of-two
    dims where an exact Hadamard doesn't exist (e.g. down_proj in=11008)."""
    g = torch.Generator(device="cpu").manual_seed(seed + n)
    A = torch.randn(n, n, generator=g, dtype=torch.float32)
    Q, R = torch.linalg.qr(A)
    Q *= torch.sign(torch.diagonal(R)).unsqueeze(0)  # fix sign ambiguity → proper rotation
    return Q.to(device=device, dtype=dtype)


def make_rotation(n: int, device, dtype=torch.float32, kind: str = "hadamard"):
    """Return an orthogonal (n×n) rotation. 'hadamard' uses Sylvester Hadamard when n is a
    power of two and falls back to random orthogonal otherwise; 'orthogonal' always random."""
    if kind == "hadamard" and _is_pow2(n):
        return hadamard_matrix(n, device, dtype)
    return random_orthogonal(n, device, dtype)


class RotatedQuantLinear(nn.Module):
    """Drop-in for nn.Linear implementing y = Wx via the rotated, quantized factorization
        y = quant_act(x @ Qᵀ) @ quant_w(W @ Qᵀ)ᵀ + bias
    Q is orthogonal so this equals Wx in fp; quantization sees the *rotated* (incoherent)
    weights/activations. Q is shared per input-dim (passed in) to bound memory."""

    def __init__(self, linear: nn.Linear, Q: torch.Tensor, w_bits: int,
                 a_bits: int | None = None):
        super().__init__()
        dtype = linear.weight.dtype
        W = linear.weight.data.float()
        Wr = W @ Q.float().t()                          # rotate input axis of the weight
        wq = quant.FakeQuant.quant_weight_per_channel(Wr, w_bits)
        self.register_buffer("weight", wq.to(dtype))
        self.register_buffer("Q", Q.to(dtype))
        self.bias = linear.bias
        self.a_bits = a_bits

    def forward(self, x):
        xr = x.to(self.Q.dtype) @ self.Q.t()           # rotate activations (exact, fp)
        if self.a_bits is not None:
            xr = quant.FakeQuant.quant_act(xr.float(), self.a_bits, mode="per_token").to(self.weight.dtype)
        return F.linear(xr, self.weight, self.bias)


class Rotation:
    """QuaRot/SpinQuant-style incoherence processing. Shares one rotation per input-dim."""
    name = "rotation"

    def __init__(self, w_bits: int = 4, a_bits: int | None = None, kind: str = "hadamard"):
        self.w_bits, self.a_bits, self.kind = w_bits, a_bits, kind

    def apply(self, model, ctx=None):
        dev = model.language_model.model.embed_tokens.weight.device
        rotations: dict[int, torch.Tensor] = {}
        n = 0
        for full, parent, attr, lin in list(quant.backbone_linears(model)):
            d = lin.in_features
            if d not in rotations:
                rotations[d] = make_rotation(d, dev, torch.float32, self.kind)
            setattr(parent, attr,
                    RotatedQuantLinear(lin, rotations[d], self.w_bits, self.a_bits).to(dev))
            n += 1
        kinds = {d: ("hadamard" if (self.kind == "hadamard" and _is_pow2(d)) else "orth")
                 for d in rotations}
        print(f"[Rotation] grafted {n} linears @ W{self.w_bits}"
              f"{'A'+str(self.a_bits) if self.a_bits else ''} | rotations={kinds}")
        return float(self.w_bits)


# ════════════════════════════════ AWQ scaling ════════════════════════════════
class AWQScaledLinear(nn.Module):
    """y = quant_act(x / s) @ quant_w(W·s)ᵀ + bias, with per-input-channel scale s.
    Scaling salient channels UP before weight quant (and dividing the activation by the
    same s) preserves their resolution under coarse weight grids."""

    def __init__(self, linear: nn.Linear, s: torch.Tensor, w_bits: int,
                 a_bits: int | None = None):
        super().__init__()
        dtype = linear.weight.dtype
        W = linear.weight.data.float()
        s = s.float().clamp_min(1e-6)
        Ws = W * s.unsqueeze(0)                         # scale columns (input channels)
        wq = quant.FakeQuant.quant_weight_per_channel(Ws, w_bits)
        self.register_buffer("weight", wq.to(dtype))
        self.register_buffer("inv_s", (1.0 / s).to(dtype))
        self.bias = linear.bias
        self.a_bits = a_bits

    def forward(self, x):
        xs = x * self.inv_s
        if self.a_bits is not None:
            xs = quant.FakeQuant.quant_act(xs.float(), self.a_bits, mode="per_token").to(self.weight.dtype)
        return F.linear(xs, self.weight, self.bias)


class AWQScale:
    """Activation-aware weight scaling. Needs ctx['act_meanabs'][full_name] (per-input-
    channel mean|x|, from the calibration buffer). α (salience exponent) grid-searched per
    layer to minimize weight-quant MSE weighted by activation energy."""
    name = "awq"

    def __init__(self, w_bits: int = 4, a_bits: int | None = None,
                 alphas=(0.0, 0.25, 0.5, 0.75, 1.0)):
        self.w_bits, self.a_bits, self.alphas = w_bits, a_bits, list(alphas)

    def _best_scale(self, W, act):
        """Pick s = act^α minimizing Σ_j act_j · ||W_:,j − dequant(W_:,j·s_j)/s_j||² ."""
        W = W.float()
        act = act.float().clamp_min(1e-6)
        act_n = act / act.mean()
        best, best_err = None, float("inf")
        for a in self.alphas:
            s = act_n.pow(a).clamp(1e-2, 1e2)
            Wq = quant.FakeQuant.quant_weight_per_channel(W * s.unsqueeze(0), self.w_bits) / s.unsqueeze(0)
            err = (((Wq - W) ** 2).mean(0) * act).sum().item()  # activation-weighted MSE
            if err < best_err:
                best_err, best = err, s
        return best

    def apply(self, model, ctx):
        act_stats = ctx["act_meanabs"]
        dev = model.language_model.model.embed_tokens.weight.device
        n = 0
        for full, parent, attr, lin in list(quant.backbone_linears(model)):
            act = act_stats.get(full)
            if act is None:
                s = torch.ones(lin.in_features)
            else:
                s = self._best_scale(lin.weight.data, act.to(lin.weight.device))
            setattr(parent, attr,
                    AWQScaledLinear(lin, s.to(dev), self.w_bits, self.a_bits).to(dev))
            n += 1
        print(f"[AWQ] grafted {n} linears @ W{self.w_bits}"
              f"{'A'+str(self.a_bits) if self.a_bits else ''} (α∈{self.alphas})")
        return float(self.w_bits)


# ═══════════════════════════ SVD low-rank residual ═══════════════════════════
class SVDResidualLinear(nn.Module):
    """y = x @ (Q(W) + UVᵀ)ᵀ + bias.  Q(W) is the b-bit per-channel weight; UVᵀ is the
    rank-r fp16 SVD of the residual R = W − Q(W). The low-rank correction cancels the
    largest structured quant-error directions for a small parameter overhead."""

    def __init__(self, linear: nn.Linear, w_bits: int, rank: int):
        super().__init__()
        dtype = linear.weight.dtype
        W = linear.weight.data.float()
        Wq = quant.FakeQuant.quant_weight_per_channel(W, w_bits)
        R = W - Wq
        U, S, Vh = torch.linalg.svd(R, full_matrices=False)
        r = min(rank, S.numel())
        U = (U[:, :r] * S[:r].unsqueeze(0))            # (out, r)  fold Σ into U
        V = Vh[:r, :]                                  # (r, in)
        self.register_buffer("weight", Wq.to(dtype))
        self.register_buffer("U", U.to(dtype))
        self.register_buffer("V", V.to(dtype))
        self.bias = linear.bias

    def forward(self, x):
        y = F.linear(x, self.weight, self.bias)
        y = y + (x @ self.V.t()) @ self.U.t()          # low-rank correction
        return y


class SVDResidual:
    name = "svd_residual"

    def __init__(self, w_bits: int = 3, rank: int = 32):
        self.w_bits, self.rank = w_bits, rank

    def apply(self, model, ctx=None):
        dev = model.language_model.model.embed_tokens.weight.device
        tot_bits, tot_w, n = 0.0, 0, 0
        for full, parent, attr, lin in list(quant.backbone_linears(model)):
            mod = SVDResidualLinear(lin, self.w_bits, self.rank).to(dev)
            setattr(parent, attr, mod)
            out_f, in_f = lin.weight.shape
            r = min(self.rank, min(out_f, in_f))
            eff = self.w_bits * out_f * in_f + 16 * r * (out_f + in_f)  # fp16 low-rank cost
            tot_bits += eff
            tot_w += out_f * in_f
            n += 1
        eff_bits = tot_bits / max(tot_w, 1)
        print(f"[SVDResidual] grafted {n} linears @ W{self.w_bits}+rank{self.rank} "
              f"→ effective {eff_bits:.2f} bits")
        return eff_bits


# ═══════════════════════════ mixed precision (per-layer) ══════════════════════
class PrecondQuantLinear(nn.Module):
    """Generalizes RotatedQuantLinear to ANY invertible preconditioner M (orthogonal or not):
        y = quant_act(x @ Mᵀ) @ quant_w(W @ M⁻¹)ᵀ + bias
    Exact in fp because Minv·M = I. M⁻¹ = Mᵀ for orthogonal transforms; for ZCA it's C^{1/2}."""
    def __init__(self, linear: nn.Linear, M: torch.Tensor, Minv: torch.Tensor,
                 w_bits: int, a_bits: int | None = None):
        super().__init__()
        dtype = linear.weight.dtype
        A = linear.weight.data.float() @ Minv.float()            # W M⁻¹
        wq = quant.FakeQuant.quant_weight_per_channel(A, w_bits)
        self.register_buffer("weight", wq.to(dtype))
        self.register_buffer("M", M.to(dtype))
        self.bias = linear.bias
        self.a_bits = a_bits

    def forward(self, x):
        z = x.to(self.M.dtype) @ self.M.t()
        if self.a_bits is not None:
            z = quant.FakeQuant.quant_act(z.float(), self.a_bits, mode="per_token").to(self.weight.dtype)
        return F.linear(z, self.weight, self.bias)


class Transform:
    """Closed-loop wrapper for the rotations.py transform zoo — graft any transform as the
    activation/weight preconditioner before W{w}A{a} quant. Data-independent transforms share
    one M per input-dim; pca/zca capture per-layer input covariance (ctx calib_frames); polar
    uses the layer weight. Lets every transform in 11_transform_compare.py be rollout-scored."""
    name = "transform"

    def __init__(self, kind: str, w_bits: int = 4, a_bits: int | None = 4):
        self.kind, self.w_bits, self.a_bits = kind, w_bits, a_bits

    def _capture_cov(self, model, ctx):
        import capture as cap_mod
        layers = {n: lin for n, _, _, lin in quant.backbone_linears(model)}
        cov = {n: torch.zeros(l.in_features, l.in_features, device=l.weight.device, dtype=torch.float32)
               for n, l in layers.items()}
        handles = []
        def mk(name):
            def hook(m, inp, out):
                x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
                cov[name].addmm_(x.t(), x)
            return hook
        for n, l in layers.items():
            handles.append(l.register_forward_hook(mk(n)))
        with torch.no_grad():
            for instr, img in ctx["calib_frames"]:
                model(**cap_mod.build_inputs(ctx["processor"], model, ctx["device"], instr, img))
        for h in handles:
            h.remove()
        return cov

    def apply(self, model, ctx=None):
        import rotations as R
        dev = model.language_model.model.embed_tokens.weight.device
        data_dep, weight_dep = self.kind in R.DATA_DEP, self.kind in R.WEIGHT_DEP
        cov = self._capture_cov(model, ctx) if data_dep else {}
        cache, n, fbs = {}, 0, 0
        for full, parent, attr, lin in list(quant.backbone_linears(model)):
            d = lin.in_features
            if data_dep:
                M, Minv, fb = R.build_transform(d, self.kind, dev, cov=cov[full])
            elif weight_dep:
                M, Minv, fb = R.build_transform(d, self.kind, dev, W=lin.weight.data)
            else:
                if d not in cache:
                    cache[d] = R.build_transform(d, self.kind, dev)
                M, Minv, fb = cache[d]
            fbs += int(fb)
            setattr(parent, attr,
                    PrecondQuantLinear(lin, M, Minv, self.w_bits, self.a_bits).to(dev))
            n += 1
        print(f"[Transform:{self.kind}] grafted {n} linears @ W{self.w_bits}"
              f"{'A'+str(self.a_bits) if self.a_bits else ''} ({fbs} pow2-fallback layers)")
        return float(self.w_bits)


class GPTQ:
    """Hessian-based error-feedback weight quant (Frantar et al. 2210.17323). Quantizes
    each backbone linear column-by-column, after each step pushing the rounding error onto
    the not-yet-quantized columns through the inverse activation Hessian H⁻¹ (H = XᵀX on the
    calibration buffer). Recovers far more than naive rounding at W3/W2.

    Needs forward passes, so it reads ctx['calib_frames'], ctx['processor'], ctx['device']
    and captures per-layer Hessians itself (one pass, hooks on every backbone linear).
    Optionally composes with a Hadamard input rotation (rotate=True) for QuaRot+GPTQ.
    """
    name = "gptq"

    def __init__(self, w_bits: int = 3, blocksize: int = 128, percdamp: float = 0.01,
                 rotate: bool = False):
        self.w_bits, self.blocksize, self.percdamp, self.rotate = w_bits, blocksize, percdamp, rotate

    def _capture_hessians(self, model, ctx):
        import capture as cap_mod
        layers = {n: lin for n, _, _, lin in quant.backbone_linears(model)}
        H = {n: torch.zeros(l.in_features, l.in_features, device=l.weight.device, dtype=torch.float32)
             for n, l in layers.items()}
        cnt = {n: 0 for n in layers}
        handles = []
        def mk(name):
            def hook(m, inp, out):
                x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])  # (tok,in)
                H[name].addmm_(x.t(), x)                                   # accumulate XᵀX
                cnt[name] += x.shape[0]
            return hook
        for n, l in layers.items():
            handles.append(l.register_forward_hook(mk(n)))
        with torch.no_grad():
            for instr, img in ctx["calib_frames"]:
                model(**cap_mod.build_inputs(ctx["processor"], model, ctx["device"], instr, img))
        for h in handles:
            h.remove()
        for n in H:
            H[n] *= 2.0 / max(cnt[n], 1)                                   # H = 2 E[xxᵀ]
        return H, layers

    def _quantize_layer(self, W, H):
        """Classic GPTQ on one weight (out,in) with Hessian H (in,in). Per-output-channel
        symmetric scale from |W| max. Returns dequantized weight (out,in)."""
        W = W.float().clone()
        out_f, in_f = W.shape
        qmax = 2 ** (self.w_bits - 1) - 1
        scale = W.abs().amax(1, keepdim=True).clamp_min(1e-8) / qmax       # (out,1) per row

        dead = torch.diag(H) == 0
        H[dead, dead] = 1.0
        W[:, dead] = 0.0
        damp = self.percdamp * torch.diag(H).mean()
        H[range(in_f), range(in_f)] += damp
        # H⁻¹ as an upper-triangular Cholesky factor (GPTQ's stable parameterization)
        L = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(L)
        Hinv = torch.linalg.cholesky(Hinv, upper=True)

        Q = torch.zeros_like(W)
        for i0 in range(0, in_f, self.blocksize):
            i1 = min(i0 + self.blocksize, in_f)
            Wb = W[:, i0:i1].clone()
            Qb = torch.zeros_like(Wb)
            Eb = torch.zeros_like(Wb)
            Hb = Hinv[i0:i1, i0:i1]
            for j in range(i1 - i0):
                w = Wb[:, j]
                d = Hb[j, j]
                q = torch.round(w / scale.squeeze(1)).clamp(-qmax - 1, qmax) * scale.squeeze(1)
                Qb[:, j] = q
                err = (w - q) / d
                Wb[:, j:] -= err.unsqueeze(1) * Hb[j, j:].unsqueeze(0)
                Eb[:, j] = err
            Q[:, i0:i1] = Qb
            W[:, i1:] -= Eb @ Hinv[i0:i1, i1:]                            # propagate to later blocks
        return Q

    def apply(self, model, ctx):
        dev = model.language_model.model.embed_tokens.weight.device
        rot = {}
        if self.rotate:
            for full, _, _, lin in quant.backbone_linears(model):
                d = lin.in_features
                if d not in rot:
                    rot[d] = make_rotation(d, dev, torch.float32, "hadamard")
        H, layers = self._capture_hessians(model, ctx)
        n = 0
        for full, parent, attr, lin in list(quant.backbone_linears(model)):
            W = lin.weight.data.float()
            Hl = H[full]
            if self.rotate:                                              # rotate input axis first
                Q = rot[lin.in_features]
                W = W @ Q.t()
                Hl = Q @ Hl @ Q.t()
            wq = self._quantize_layer(W, Hl)
            if self.rotate:
                mod = RotatedQuantLinearPrequant(lin, rot[lin.in_features], wq).to(dev)
            else:
                mod = quant.FakeQuant.Linear(lin, weight_dq=wq).to(dev)
            setattr(parent, attr, mod)
            del H[full]
            n += 1
        print(f"[GPTQ] quantized {n} linears @ W{self.w_bits}"
              f"{' +Hadamard' if self.rotate else ''} (blocksize {self.blocksize})")
        return float(self.w_bits)


class RotatedQuantLinearPrequant(nn.Module):
    """Like RotatedQuantLinear but takes an already-rotated, already-quantized weight (used
    by GPTQ+rotation, which quantizes in the rotated basis): y = (x @ Qᵀ) @ wqᵀ + bias."""
    def __init__(self, linear: nn.Linear, Q: torch.Tensor, weight_dq: torch.Tensor):
        super().__init__()
        dtype = linear.weight.dtype
        self.register_buffer("weight", weight_dq.to(dtype))
        self.register_buffer("Q", Q.to(dtype))
        self.bias = linear.bias

    def forward(self, x):
        return F.linear(x.to(self.Q.dtype) @ self.Q.t(), self.weight, self.bias)


class MixedPrecision:
    """Per-layer W4/W8 (configurable pair) allocated to a target average bit-width by an
    activation-scaled weight-quant-error sensitivity. Layers whose quant error matters most
    (large error × large activation energy) get the high bit-width."""
    name = "mixed"

    def __init__(self, target_bits: float = 5.0, lo: int = 4, hi: int = 8):
        self.target_bits, self.lo, self.hi = target_bits, lo, hi

    def _sensitivity(self, W, act):
        W = W.float()
        qmax = 2 ** (self.lo - 1) - 1
        s = W.abs().amax(1, keepdim=True).clamp_min(1e-8) / qmax
        Wq = torch.round(W / s).clamp(-qmax - 1, qmax) * s
        per_out_err = (W - Wq).pow(2).mean(1)          # (out,)
        a = 1.0 if act is None else act.float().mean().item()
        return per_out_err.mean().item() * a           # scalar layer sensitivity

    def apply(self, model, ctx=None):
        act_stats = (ctx or {}).get("act_meanabs", {})
        layers = list(quant.backbone_linears(model))
        sens = []
        for full, parent, attr, lin in layers:
            sens.append((full, self._sensitivity(lin.weight.data, act_stats.get(full))))
        # rank by sensitivity; promote the most sensitive layers to hi until budget spent
        order = sorted(range(len(layers)), key=lambda i: sens[i][1], reverse=True)
        sizes = [layers[i][3].weight.numel() for i in range(len(layers))]
        total = sum(sizes)
        budget_hi = max(0.0, (self.target_bits - self.lo) / (self.hi - self.lo)) * total
        assigned, used_hi = {}, 0.0
        for i in order:
            if used_hi + sizes[i] <= budget_hi:
                assigned[i] = self.hi
                used_hi += sizes[i]
            else:
                assigned[i] = self.lo
        dev = model.language_model.model.embed_tokens.weight.device
        tot_bits = 0
        for i, (full, parent, attr, lin) in enumerate(layers):
            b = assigned[i]
            setattr(parent, attr, quant.FakeQuant.Linear(lin, w_bits=b).to(dev))
            tot_bits += b * sizes[i]
        eff = tot_bits / total
        nhi = sum(1 for v in assigned.values() if v == self.hi)
        print(f"[MixedPrecision] {nhi}/{len(layers)} layers @ W{self.hi}, rest @ W{self.lo} "
              f"→ effective {eff:.2f} bits (target {self.target_bits})")
        return eff
