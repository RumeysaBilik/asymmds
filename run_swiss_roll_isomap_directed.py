#!/usr/bin/env python3
"""
run_swiss_roll_isomap_directed.py -- swiss roll, distance matrix built with
classical Isomap's own recipe (k-NN graph + Dijkstra), but kept DIRECTED
instead of forcing the usual symmetric result.

Why this can still be asymmetric with zero Randers field: sklearn's
kneighbors_graph is a directed adjacency by construction -- j being in i's
k nearest neighbours does NOT imply i is in j's. Classical Isomap throws
that asymmetry away (undirected Dijkstra -> symmetric D). Here we keep
directed=True, so D_asym[i,j] != D_asym[j,i] in general, coming purely from
k-NN-membership asymmetry -- no omega field, no isumap-style local
normalisation/star-graph/t-conorm (contrast with run_swiss_roll_isumap.py).

t and alpha(t) ground truth are still known for swiss roll, so
direction_accuracy_swiss (test.py) can check whether this purely
combinatorial asymmetry happens to correlate with the true Randers field at
all -- that's an open empirical question, not assumed.

Usage
-----
    python3 run_swiss_roll_isomap_directed.py
    python3 run_swiss_roll_isomap_directed.py --n 2000 --epochs 500
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
from randers_bridge import compute_dist_matrix
from randers_umap import randers_umap_fit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--k", type=int, default=15, help="k-NN for the geodesic backbone")
    p.add_argument("--emb-k", type=int, default=20, help="n_neighbors for randers_umap_fit")
    p.add_argument("--neg", type=int, default=10)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--gravity", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="swiss_embedding_isomap_directed")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    if not args.quiet:
        print(f"Generating swiss roll: n={args.n}")
    X, omega, t = make_swiss_roll_randers(args.n, seed=42)

    if not args.quiet:
        print(f"\nBuilding directed Isomap-style D (k-NN + directed Dijkstra, no omega)...")
    D_asym, _ = compute_dist_matrix(X, n_neighbors=args.k, path_method="auto",
                                    randers_field=None, directed=True)
    if not args.quiet:
        print(f"D_asym: {D_asym.shape}  symmetric={np.allclose(D_asym, D_asym.T)}")

    out = randers_umap_fit(D_asym, n_neighbors=args.emb_k, n_negative_samples=args.neg,
                            n_epochs=args.epochs, use_drift=True,
                            use_gravity=args.gravity, seed=args.seed,
                            verbose=not args.quiet)
    Y, B = out["Y"], out["B"]

    # ---- plot --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 8))
    sc = ax.scatter(Y[:, 0], Y[:, 1], c=t, cmap="viridis", s=10, alpha=0.85, linewidths=0)
    plt.colorbar(sc, ax=ax, label="t (intrinsic coordinate)")

    bn = np.linalg.norm(B, axis=1)
    big = np.argsort(bn)[::-1][:25]
    if bn.max() > 0:
        sc_scale = 0.12 * (Y.max() - Y.min()) / bn.max()
        ax.quiver(Y[big, 0], Y[big, 1], B[big, 0] * sc_scale, B[big, 1] * sc_scale,
                  color="k", alpha=0.6, width=0.004, scale=1, scale_units="xy")

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Randers-UMAP, directed-Isomap D (no omega)  (n={args.n})", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=150)

    np.savez(f"{args.out}.npz", Y=Y, B=B, t=t, X=X, omega=omega)

    if not args.quiet:
        print(f"\nwrote {args.out}.png and {args.out}.npz")


if __name__ == "__main__":
    main()
