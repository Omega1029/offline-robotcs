#!/usr/bin/env python3
"""
13_rotation_overhead.py — the deployment cost of the rotation step, per transform CLASS.

The quantized weights are identical across transforms (all W4); what differs is how the
input rotation Qx is applied per backbone linear, every forward. Two regimes:
  structured (Hadamard/DCT/SRHT/...): fast O(n log n) transform, matrix-free  -> cheap
  dense      (random-orth/PCA/ZCA/...): O(n^2) matmul + stored n x n matrix    -> expensive

We microbenchmark the TOTAL rotation cost of one policy query (the rotation applied to every
backbone linear's input activation), for: identity (no-op), FWHT (fast structured), and dense
matmul (generic orthogonal). Activation token count and the per-dim layer counts match the
Llama-2-7B backbone: 192 layers with in=4096 (q,k,v,o,gate,up), 32 with in=11008 (down).

Run:  CUDA_VISIBLE_DEVICES=0 ../venv/bin/python 13_rotation_overhead.py
"""
import argparse, time
import torch

# Llama-2-7B backbone: (input_dim, num_layers_with_that_input)
LAYERS = [(4096, 192), (11008, 32)]
TOKENS = 380  # ~one policy-query sequence length


def fwht(x):
    """Fast Walsh-Hadamard transform along the last axis (n must be a power of two).
    O(n log n), matrix-free."""
    orig = x.shape; n = orig[-1]
    x = x.clone()
    h = 1
    while h < n:
        x = x.view(*orig[:-1], n // (2 * h), 2, h)
        a = x[..., 0, :].clone(); b = x[..., 1, :].clone()
        x[..., 0, :] = a + b
        x[..., 1, :] = a - b
        x = x.reshape(*orig)
        h *= 2
    return x / (n ** 0.5)


def is_pow2(n): return n > 0 and (n & (n - 1)) == 0


@torch.no_grad()
def timeit(fn, iters, device):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    s = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - s) / iters * 1e3  # ms per call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    dev = torch.device("cuda:0")
    print(f"device={torch.cuda.get_device_name(0)} dtype={args.dtype} tokens={TOKENS}\n")

    # per-query totals (sum over all layers), and per-dim matrix memory
    totals = {"identity": 0.0, "fwht/structured": 0.0, "dense matmul": 0.0}
    dense_mem = 0
    for n, nlayers in LAYERS:
        x = torch.randn(TOKENS, n, device=dev, dtype=dt)
        M = torch.randn(n, n, device=dev, dtype=dt)
        dense_mem += M.numel() * M.element_size()  # one shared matrix per dim

        t_id = timeit(lambda: x.clone(), args.iters, dev)
        t_dense = timeit(lambda: x @ M.t(), args.iters, dev)
        if is_pow2(n):
            t_struct = timeit(lambda: fwht(x.float()).to(dt), args.iters, dev)
            struct_label = "FWHT"
        else:
            t_struct = t_dense; struct_label = "dense(no pow2 FWHT)"

        print(f"dim {n:5d} x {nlayers:3d} layers | identity {t_id:6.3f}ms  "
              f"{struct_label} {t_struct:6.3f}ms  dense {t_dense:6.3f}ms  (per layer-call)")
        totals["identity"] += t_id * nlayers
        totals["fwht/structured"] += t_struct * nlayers
        totals["dense matmul"] += t_dense * nlayers

    print("\n== per-policy-query rotation overhead (sum over all 224 backbone linears) ==")
    for k, v in totals.items():
        print(f"  {k:18s} {v:8.2f} ms")
    print(f"\n  dense rotation-matrix storage (shared per dim): {dense_mem/1e6:.0f} MB "
          f"(structured/FWHT: 0 MB, matrix-free)")
    print("\nTakeaway: structured transforms (DCT/Hadamard via FWHT/FFT) add a small, matrix-free\n"
          "rotation cost; dense transforms (random-orth/PCA/ZCA) pay an O(n^2) matmul AND store\n"
          "the matrices -- so among the accuracy-tied global mixers, the STRUCTURED ones are the\n"
          "only practically deployable choice.")


if __name__ == "__main__":
    main()
