#!/usr/bin/env python3
"""
run_swiss_roll.py -- Randers-UMAP on the swiss-roll + Randers field dataset,
drift (B) initialised via the "locate the target, embed it" method.

    b_i^virtual target := x_i + omega_i                (real ambient point)
    embed n real + n virtual points together, plain UMAP
    b_i := y_i^virtual - y_i                            (in the embedding)
    re-embed the n real points with that b_i frozen + attached (B_fixed)

Usage
-----
    bash get_data.sh not needed -- swiss roll is generated, not loaded
    python run_swiss_roll.py
    python run_swiss_roll.py --n 2000 --locate-epochs 200 --epochs 500

Outputs
-------
    <out>.png   embedding coloured by t, drift arrows on the top-N
    <out>.npz   Y, B, t, X, omega
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

from randers_bridge import compute_dist_matrix
from randers_umap import randers_umap_fit


def make_swiss_roll_randers(n, seed=42):
    """Exact construction from generated_swiss_roll-2.py, parameterised by n."""
    rng = np.random.RandomState(seed)

    t = 1.5 * np.pi * (1 + 2 * rng.rand(n))
    height = 21 * rng.rand(n)

    x = t * np.cos(t)
    z = t * np.sin(t)
    y = height
    X = np.column_stack([x, y, z])

    V = np.column_stack([-X[:, 2], np.zeros(n), X[:, 0]])
    V = V / np.linalg.norm(V, axis=1)[:, None]

    alpha = -0.5 * np.cos(t) + 0.3 * np.sin(t)
    omega = alpha[:, None] * V

    return X, omega, t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n",      type=int, default=1000)
    p.add_argument("--k",      type=int, default=15, help="k-NN for the geodesic backbone")
    p.add_argument("--emb-k",  type=int, default=20, help="n_neighbors for randers_umap_fit")
    p.add_argument("--neg",    type=int, default=10)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--locate-epochs", type=int, default=500)
    p.add_argument("--clip-delta", type=float, default=0.01)
    p.add_argument("--seed",   type=int, default=0)
    p.add_argument("--out",    default="swiss_embedding")
    p.add_argument("--quiet",  action="store_true")
    args = p.parse_args()

    # ---- data --------------------------------------------------------------
    if not args.quiet:
        print(f"Generating swiss roll: n={args.n}")
    X, omega, t = make_swiss_roll_randers(args.n, seed=42)
    n = args.n

    # ---- locate: embed n real + n virtual (x_i+omega_i) points together ----
    X_virtual = X + omega
    X_aug = np.vstack([X, X_virtual])

    if not args.quiet:
        print(f"\nLocate: building symmetric geodesic D on {2*n} augmented points...")
    D_sym_aug, _ = compute_dist_matrix(X_aug, n_neighbors=args.k, path_method="auto",
                                       randers_field=None)
    out1 = randers_umap_fit(D_sym_aug, n_neighbors=args.emb_k, n_negative_samples=args.neg,
                            n_epochs=args.locate_epochs, use_drift=False,
                            seed=args.seed, verbose=not args.quiet)
    Y_real0, Y_virtual0 = out1["Y"][:n], out1["Y"][n:]

    B = Y_virtual0 - Y_real0
    limit = 1.0 - args.clip_delta
    bn0 = np.linalg.norm(B, axis=1, keepdims=True)
    B = B * np.minimum(1.0, limit / np.maximum(bn0, 1e-12))

    if not args.quiet:
        bn = np.linalg.norm(B, axis=1)
        print(f"B located: mean||b||={bn.mean():.4f}  max||b||={bn.max():.4f}  "
              f"clipped={(bn >= limit - 1e-9).sum()}/{n}")

    # ---- apply: real D_asym, B frozen + attached ----------------------------
    if not args.quiet:
        print(f"\nApply: building asymmetric D_asym on the {n} real points...")
    D_asym, _ = compute_dist_matrix(X, n_neighbors=args.k, path_method="auto",
                                    randers_field=omega)
    out2 = randers_umap_fit(D_asym, n_neighbors=args.emb_k, n_negative_samples=args.neg,
                            n_epochs=args.epochs, use_drift=True,
                            B_fixed=B, Y_init_override=Y_real0,
                            clip_delta=args.clip_delta, seed=args.seed, verbose=not args.quiet)
    Y, B = out2["Y"], out2["B"]

    if not args.quiet:
        bn = np.linalg.norm(B, axis=1)
        print(f"\nextent={Y.max()-Y.min():.2f}  mean||b||={bn.mean():.4f}  "
              f"max||b||={bn.max():.4f}  clipped={(bn >= limit - 1e-9).sum()}/{n}")

    # ---- plot ----------------------------------------------------------------
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
    ax.set_title(f"Randers-UMAP swiss-roll, located-drift init  (n={n})", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=150)

    np.savez(f"{args.out}.npz", Y=Y, B=B, t=t, X=X, omega=omega)

    if not args.quiet:
        print(f"\nwrote {args.out}.png and {args.out}.npz")


if __name__ == "__main__":
    main()
