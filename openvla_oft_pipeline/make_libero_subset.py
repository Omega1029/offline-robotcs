#!/usr/bin/env python3
"""Extract a small, representative subset of LIBERO demos for on-device (Jetson) eval.

Takes N evenly-spaced demos from every task .hdf5 (covers all 40 tasks), preserving
`data.attrs` (language/bddl/env_args) and per-demo attrs (init_state). Demos are
renumbered demo_0..demo_{N-1} so standard LIBERO loaders see a contiguous set.
"""
import argparse, sys
from pathlib import Path
import h5py

def pick(n_have, n_want):
    if n_want >= n_have:
        return list(range(n_have))
    step = n_have / n_want
    return sorted({int(i * step) for i in range(n_want)})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/justin_williams1/OfflineAutonomy/offline-robotcs/datasets/libero")
    ap.add_argument("--dst", default="/home/justin_williams1/OfflineAutonomy/offline-robotcs/datasets/libero_subset")
    ap.add_argument("--demos_per_task", type=int, default=1)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    files = sorted(src.rglob("*.hdf5"))
    if not files:
        sys.exit(f"No .hdf5 under {src}")
    print(f"Found {len(files)} task files; taking {args.demos_per_task} demo(s) each")

    total_kept = 0
    for f in files:
        rel = f.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(f, "r") as h, h5py.File(out, "w") as o:
            d = h["data"]
            demos = sorted(d.keys(), key=lambda k: int(k.split("_")[1]))
            idxs = pick(len(demos), args.demos_per_task)
            og = o.create_group("data")
            for k, v in d.attrs.items():
                og.attrs[k] = v
            og.attrs["num_demos"] = str(len(idxs))  # reflect the trimmed count
            for new_i, di in enumerate(idxs):
                d.copy(d[demos[di]], og, name=f"demo_{new_i}")  # deep copy + attrs
            total_kept += len(idxs)
        print(f"  {rel}  ->  {len(idxs)} demos")
    print(f"\nDone. {total_kept} demos written to {dst}")

if __name__ == "__main__":
    main()
