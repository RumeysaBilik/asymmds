#!/usr/bin/env python3
"""
plot_x_virtual.py -- diagnostic only, does not touch run_swiss_roll.py.

Visualises X (real swiss-roll points) vs X_virtual = X + omega (the
"imaginary target" points used to initialise the drift), both in the same
ambient R^3, plus a subsample of connecting segments X_i -> X_virtual_i so
the offset direction/magnitude can be inspected directly.

Usage
-----
    python plot_x_virtual.py --n 2000 --n-lines 60
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_swiss_roll import make_swiss_roll_randers


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--n-lines", type=int, default=60, help="how many X_i->X_virtual_i segments to draw")
    p.add_argument("--out", default="x_virtual_check")
    args = p.parse_args()

    X, omega, t = make_swiss_roll_randers(args.n, seed=42)
    X_virtual = X + omega

    off = np.linalg.norm(X_virtual - X, axis=1)
    print(f"n={args.n}")
    print(f"||X_virtual - X|| : min={off.min():.4f}  median={np.median(off):.4f}  max={off.max():.4f}")

    # 1-NN spacing of X itself, for scale reference
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    d, _ = nn.kneighbors(X)
    nn1 = d[:, 1]
    print(f"1-NN spacing in X  : median={np.median(nn1):.4f}")
    print(f"offset / 1-NN spacing ratio: median={np.median(off)/np.median(nn1):.3f}")

    fig = plt.figure(figsize=(11, 11))
    ax = fig.add_subplot(111, projection="3d")

    X_ctr = X - X.mean(axis=0)
    Xv_ctr = X_virtual - X.mean(axis=0)

    ax.scatter(X_ctr[:, 0], X_ctr[:, 1], X_ctr[:, 2], c=t, cmap="viridis",
              s=4, alpha=0.6, label="X (real)")
    ax.scatter(Xv_ctr[:, 0], Xv_ctr[:, 1], Xv_ctr[:, 2], c="red",
              s=4, alpha=0.5, label="X_virtual = X + omega")

    rng = np.random.default_rng(0)
    idx = rng.choice(args.n, size=min(args.n_lines, args.n), replace=False)
    for i in idx:
        ax.plot([X_ctr[i, 0], Xv_ctr[i, 0]],
               [X_ctr[i, 1], Xv_ctr[i, 1]],
               [X_ctr[i, 2], Xv_ctr[i, 2]], color="k", lw=0.7, alpha=0.6)

    ax.set_title(f"X vs X_virtual=X+omega  (n={args.n}, {args.n_lines} sample segments shown)")
    ax.legend()
    fig.savefig(f"{args.out}.png", dpi=150)
    print(f"wrote {args.out}.png")


if __name__ == "__main__":
    main()
