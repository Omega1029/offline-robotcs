#!/usr/bin/env python3
"""
rotations.py — a zoo of orthogonal / invertible transforms to precondition activations (and
weights) before low-bit quantization. All plug into the same factorization used by rot_*_had:

    y = W x = (W M⁻¹)(M x)          # exact for any invertible M

We quantize A = W M⁻¹ (per-output-channel) and z = M x (per-token). For ORTHOGONAL M,
M⁻¹ = Mᵀ and error norms are basis-invariant (the clean QuaRot case). For non-orthogonal
preconditioners (ZCA whitening), M⁻¹ ≠ Mᵀ and we trade activation conditioning against
weight conditioning (the full-matrix generalization of AWQ/SmoothQuant).

`build_transform(n, kind, ...) -> (M, Minv, fellback)` returns the (n×n) transform. Data-
independent kinds need only n; data-dependent kinds need `cov` (input covariance, n×n) or
`W` (the layer weight). Power-of-two-only kinds fall back to a random orthogonal on non-2^k
dims (e.g. down_proj in=11008) and set fellback=True.

STARRED = the transforms called out as most relevant (fast structured + data-optimal).
ALL     = the full zoo.
"""
from __future__ import annotations
import math
import torch

STARRED = ["hadamard", "dct", "srht", "butterfly", "pca", "zca", "random_orthogonal", "identity"]
ALL = STARRED + ["dst", "hartley", "haar", "givens", "householder",
                 "permutation", "polar", "walsh"]
DATA_DEP = {"pca", "zca"}          # need input covariance
WEIGHT_DEP = {"polar"}             # need the layer weight
POW2_ONLY = {"hadamard", "walsh", "haar", "srht", "butterfly"}


def _is_pow2(n): return n > 0 and (n & (n - 1)) == 0


# ───────────────────────── data-independent builders ─────────────────────────
def _hadamard(n):
    if not _is_pow2(n): return None
    H = torch.ones(1, 1); base = torch.tensor([[1., 1.], [1., -1.]])
    for _ in range(int(round(math.log2(n)))):
        H = torch.kron(H, base)
    return H / math.sqrt(n)


def _dct(n):
    k = torch.arange(n).float().view(-1, 1); j = torch.arange(n).float().view(1, -1)
    M = torch.cos(math.pi / n * (j + 0.5) * k) * math.sqrt(2.0 / n)
    M[0] *= 1 / math.sqrt(2)                       # DCT-II, orthonormal
    return M


def _dst(n):
    k = torch.arange(1, n + 1).float().view(-1, 1); j = torch.arange(1, n + 1).float().view(1, -1)
    return torch.sin(math.pi / (n + 1) * j * k) * math.sqrt(2.0 / (n + 1))   # DST-I orthonormal


def _hartley(n):
    k = torch.arange(n).float().view(-1, 1); j = torch.arange(n).float().view(1, -1)
    ang = 2 * math.pi * k * j / n
    return (torch.cos(ang) + torch.sin(ang)) / math.sqrt(n)                  # DHT (cas), orthogonal


def _haar(n):
    if not _is_pow2(n): return None
    h = torch.tensor([[1.]])
    while h.shape[0] < n:
        m = h.shape[0]
        top = torch.kron(h, torch.tensor([[1., 1.]]))
        bot = torch.kron(torch.eye(m), torch.tensor([[1., -1.]]))
        h = torch.cat([top, bot], 0)
    return h / h.norm(dim=1, keepdim=True)                                   # row-normalized → orthonormal


def _srht(n, seed=0):
    H = _hadamard(n)
    if H is None: return None
    g = torch.Generator().manual_seed(seed + n)
    s = torch.randint(0, 2, (n,), generator=g).float() * 2 - 1
    return H * s.view(1, -1)                                                 # H·diag(±1), orthogonal


def _butterfly(n, seed=0):
    if not _is_pow2(n): return None
    g = torch.Generator().manual_seed(seed + n)
    M = torch.eye(n); logn = int(round(math.log2(n)))
    for stage in range(logn):
        stride = 1 << stage
        B = torch.eye(n)
        ang = torch.rand((n // 2,), generator=g) * 2 * math.pi
        idx = 0
        for start in range(0, n, 2 * stride):
            for off in range(stride):
                i, j = start + off, start + off + stride
                c, s = math.cos(ang[idx]), math.sin(ang[idx]); idx += 1
                B[i, i] = c; B[i, j] = -s; B[j, i] = s; B[j, j] = c
        M = B @ M
    return M


def _householder(n, k=24, seed=0):
    g = torch.Generator().manual_seed(seed + n); M = torch.eye(n)
    for _ in range(k):
        v = torch.randn(n, generator=g); v = v / v.norm()
        M = M - 2 * torch.outer(M @ v, v)                                   # apply reflection on the right
    return M


def _givens(n, sweeps=3, seed=0):
    g = torch.Generator().manual_seed(seed + n); M = torch.eye(n)
    num = sweeps * n
    pairs = torch.randint(0, n, (num, 2), generator=g)
    angs = torch.rand((num,), generator=g) * 2 * math.pi
    for t in range(num):
        i, j = int(pairs[t, 0]), int(pairs[t, 1])
        if i == j: continue
        c, s = math.cos(float(angs[t])), math.sin(float(angs[t]))
        ci, cj = M[:, i].clone(), M[:, j].clone()
        M[:, i] = c * ci - s * cj; M[:, j] = s * ci + c * cj
    return M


def _permutation(n, seed=0):
    g = torch.Generator().manual_seed(seed + n)
    return torch.eye(n)[torch.randperm(n, generator=g)]


def _random_orthogonal(n, seed=0):
    g = torch.Generator().manual_seed(seed + n)
    A = torch.randn(n, n, generator=g)
    Q, R = torch.linalg.qr(A)
    return Q * torch.sign(torch.diagonal(R)).unsqueeze(0)


# ───────────────────────── data-dependent builders ───────────────────────────
def _pca(cov):
    """Rotate into the eigenbasis of the input covariance (orthogonal, decorrelating)."""
    w, V = torch.linalg.eigh(cov.double())
    V = V.flip(1).float()                                  # high-variance first (cosmetic)
    return V.t(), V                                        # M = Vᵀ, Minv = V


def _zca(cov, eps=1e-3):
    """ZCA whitening preconditioner C^{-1/2} (symmetric PSD, NOT orthogonal). Decorrelates
    AND equalizes variance — the full-matrix generalization of per-channel smoothing."""
    w, V = torch.linalg.eigh(cov.double())
    damp = eps * w.clamp_min(0).mean()
    inv_sqrt = (w + damp).clamp_min(1e-12).rsqrt()
    sqrt = (w + damp).clamp_min(1e-12).sqrt()
    M = (V * inv_sqrt) @ V.t()                            # C^{-1/2}
    Minv = (V * sqrt) @ V.t()                             # C^{1/2}
    return M.float(), Minv.float()


def _polar(W):
    """Align input axes with the weight's right-singular basis (data-dependent on W)."""
    Wf = W.float()
    # right singular vectors V of W (in-space orthonormal)
    _, _, Vh = torch.linalg.svd(Wf, full_matrices=True)
    V = Vh.t()                                            # (in, in)
    return V.t(), V


# ───────────────────────────── dispatcher ────────────────────────────────────
_BUILDERS = {
    "identity": lambda n, **kw: torch.eye(n),
    "permutation": lambda n, seed=0, **kw: _permutation(n, seed),
    "hadamard": lambda n, **kw: _hadamard(n),
    "walsh": lambda n, **kw: _hadamard(n),          # sequency reorder ≈ same outlier behavior
    "dct": lambda n, **kw: _dct(n),
    "dst": lambda n, **kw: _dst(n),
    "hartley": lambda n, **kw: _hartley(n),
    "haar": lambda n, **kw: _haar(n),
    "slant": lambda n, **kw: _slant(n),
    "srht": lambda n, seed=0, **kw: _srht(n, seed),
    "butterfly": lambda n, seed=0, **kw: _butterfly(n, seed),
    "householder": lambda n, seed=0, **kw: _householder(n, seed=seed),
    "givens": lambda n, seed=0, **kw: _givens(n, seed=seed),
    "random_orthogonal": lambda n, seed=0, **kw: _random_orthogonal(n, seed),
}


def build_transform(n, kind, device="cpu", dtype=torch.float32, cov=None, W=None, seed=0):
    """Return (M, Minv, fellback). M, Minv are (n×n) on `device`. Orthogonal kinds set
    Minv = Mᵀ. Pow2-only kinds on non-2^k dims fall back to random orthogonal."""
    fellback = False
    if kind in DATA_DEP:
        assert cov is not None, f"{kind} needs cov"
        M, Minv = _pca(cov) if kind == "pca" else _zca(cov)
    elif kind in WEIGHT_DEP:
        assert W is not None, f"{kind} needs W"
        M, Minv = _polar(W)
    else:
        M = _BUILDERS[kind](n, seed=seed)
        if M is None:                                    # pow2-only on a non-pow2 dim
            M = _random_orthogonal(n, seed); fellback = True
        Minv = M.t()
    return M.to(device, dtype), Minv.to(device, dtype), fellback
