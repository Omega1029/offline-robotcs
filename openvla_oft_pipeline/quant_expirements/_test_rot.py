import torch, rotations as R
torch.manual_seed(0)
n = 256
cov = (lambda X: X.t() @ X)(torch.randn(4000, n)); W = torch.randn(n, n)
print(f"{'kind':18s} {'MMinv-I':>10s} {'MMt-I(orth)':>11s} {'fellback':>8s}")
for k in R.ALL:
    M, Minv, fb = R.build_transform(n, k, cov=cov, W=W)
    print(f"{k:18s} {(M@Minv-torch.eye(n)).abs().max():10.1e} {(M@M.t()-torch.eye(n)).abs().max():11.1e} {str(fb):>8s}")
print("\nnon-pow2 n=11008 structured fallback (shape+flag):")
for k in ["hadamard", "srht", "butterfly", "dct", "hartley"]:
    M, _, fb = R.build_transform(11008, k)
    print(f"  {k:10s} {tuple(M.shape)} fellback={fb}")
