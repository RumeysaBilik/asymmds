#!/usr/bin/env python3
"""
run_swiss_roll_svd_radar.py -- swiss roll, embedded via SVD-RADAR Y-init
(svd_radar.py::svd_init) + virtual-point-located B, feeding a force-directed
UMAP update (randers_umap.py::randers_umap_fit) -- NOT the Adam/Finsler-loss
loop.

[CHANGED 2026-08-12, per explicit user request] Previously this script used
finsler_mds_joint.py::finsler_mds_freeB -- SVD init followed by Adam
gradient descent against the Finsler reconstruction loss (F(Y,B)-D_asym)^2,
i.e. the "D mentality" (global, one-shot matrix regression). The actual
goal was always to use the SAME force-directed-graph update mechanism as
run_swiss_roll.py / run_swiss_roll_isumap.py (UMAP-style attraction/
repulsion, "g mentality"), just swapping WHICH method initialises Y:

    run_swiss_roll.py        -- Y_init AND B both from the virtual-point
                                 locate step (embed n real + n virtual
                                 together, B_located = Y_virtual - Y_real)
    run_swiss_roll_isumap.py -- B from SVD-of-Delta; Y from
                                 distance_graph_generation's own D (no
                                 locate step -- isumap has no omega/virtual
                                 points to locate against)
    this file                -- Y_init from SVD-RADAR on D_asym directly;
                                 B STILL from the virtual-point locate step
                                 (same mechanism as run_swiss_roll.py --
                                 B needs the true omega field to locate
                                 against, SVD-of-Delta is a different,
                                 not-yet-comparable signal)

So the only thing that differs between this script and run_swiss_roll.py
is the SOURCE of Y's starting position -- everything else (B's derivation,
the force-directed update mechanism, ramp behaviour) is identical, making
this a clean, single-variable comparison: does an SVD-derived Y_init give a
better/worse starting layout than the locate step's own Y_real0?

finsler_mds_joint.py / finsler_mds_freeB itself is untouched and still
importable for anyone who wants the Adam/Finsler-loss variant -- just no
longer wired into this script's default path.

Recipe
------
    D_asym   = compute_dist_matrix(X, randers_field=omega)
    Y_init   = svd_init(D_asym, ...)[:, :d]                  (svd_radar.py)
    B_located = locate step (virtual points x_i+omega_i, plain UMAP,
                B_located := Y_virtual - Y_real)              (same as
                                                     run_swiss_roll.py)
    Y, B     = randers_umap_fit(D_asym, Y_init_override=Y_init,
                                 B_fixed=B_located, ...)      (force-directed)

Usage
-----
    python3 run_swiss_roll_svd_radar.py
    python3 run_swiss_roll_svd_radar.py --n 2000 --svd-k 10 --epochs 500
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
from svd_radar import svd_init


def build_svd_Y_init(D_asym, svd_k=10, d=2, normalize=True):
    """
    SVD-RADAR Y-ONLY init, straight from D_asym -- the position signal from
    svd_init's [outgoing | incoming] decomposition, truncated to d dims.
    B is NOT derived here (see locate_B below) -- svd_init(Delta) would give
    a different asymmetric signal than the virtual-point locate method, and
    we want to isolate Y's init method as the only variable vs
    run_swiss_roll.py, not also swap out how B is computed.

    normalize : scale-only (divide by std, no mean subtraction) rescaling
        of D_asym before the SVD -- same recipe finsler_mds_joint.py used
        for its own normalize_D, keeps Y_init's scale sensible.
    """
    D_train = D_asym / (float(np.std(D_asym)) + 1e-8) if normalize else D_asym
    Y_init = svd_init(D_train, k=svd_k, normalize=False)[:, :d]
    return Y_init


def locate_B(X, omega, k=15, emb_k=20, neg=10, locate_epochs=500,
             clip_delta=0.01, seed=0, verbose=True):
    """
    Virtual-point locate step for B -- IDENTICAL mechanism to
    run_swiss_roll.py's run_located_drift(): embed n real + n virtual
    (x_i+omega_i) points together with plain (drift-off) UMAP, then
    B_located := Y_virtual - Y_real (in that shared embedding).
    """
    n = X.shape[0]
    X_virtual = X + omega
    X_aug = np.vstack([X, X_virtual])

    if verbose:
        print(f"\nLocate: building symmetric geodesic D on {2*n} augmented points...")
    D_sym_aug, _ = compute_dist_matrix(X_aug, n_neighbors=k, path_method="auto",
                                       randers_field=None)
    out1 = randers_umap_fit(D_sym_aug, n_neighbors=emb_k, n_negative_samples=neg,
                            n_epochs=locate_epochs, use_drift=False,
                            seed=seed, verbose=verbose)
    Y_real0, Y_virtual0 = out1["Y"][:n], out1["Y"][n:]

    B_located = Y_virtual0 - Y_real0
    limit = 1.0 - clip_delta
    bn0 = np.linalg.norm(B_located, axis=1, keepdims=True)
    B_located = B_located * np.minimum(1.0, limit / np.maximum(bn0, 1e-12))

    if verbose:
        bn = np.linalg.norm(B_located, axis=1)
        print(f"B located: mean||b||={bn.mean():.4f}  max||b||={bn.max():.4f}  "
              f"clipped={(bn >= limit - 1e-9).sum()}/{n}")

    return B_located


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--k", type=int, default=15, help="k-NN for the geodesic backbone (compute_dist_matrix)")
    p.add_argument("--emb-k", type=int, default=20, help="n_neighbors for randers_umap_fit (locate AND apply)")
    p.add_argument("--neg", type=int, default=10)
    p.add_argument("--svd-k", type=int, default=10, help="number of singular values kept in svd_init (Y_init only)")
    p.add_argument("--d", type=int, default=2, help="embedding dimension")
    p.add_argument("--clip-delta", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=500, help="force-directed update epochs (apply step)")
    p.add_argument("--locate-epochs", type=int, default=500, help="epochs for the virtual-point locate step (B)")
    p.add_argument("--gravity", action="store_true",
                    help="add per-node gravity = b_i to the update (no extra scaling)")
    p.add_argument("--ramp", action="store_true",
                    help="ramp B_located's magnitude 0->1 over the first 70%% of apply-step "
                         "epochs instead of full strength from epoch 0 (default: off, "
                         "matching run_swiss_roll.py's convention).")
    p.add_argument("--no-normalize-D", action="store_true",
                    help="skip the scale normalisation of D_asym before the SVD Y-init "
                         "(default: normalize on)")
    p.add_argument("--snapshot-every", type=int, default=None,
                    help="if given, also save <out>_snapshots.png: the apply-step "
                         "embedding every N epochs (from SVD Y_init to final), side by side")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="swiss_embedding_svd_radar")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    if not args.quiet:
        print(f"Generating swiss roll: n={args.n}")
    X, omega, t = make_swiss_roll_randers(args.n, seed=42)

    if not args.quiet:
        print(f"Building D_asym (compute_dist_matrix, true omega field)...")
    D_asym, _ = compute_dist_matrix(X, n_neighbors=args.k, path_method="auto", randers_field=omega)
    if not args.quiet:
        print(f"D_asym: {D_asym.shape}  symmetric={np.allclose(D_asym, D_asym.T)}")

    # ---- Y_init: SVD-RADAR on D_asym directly (the one thing this script
    # varies vs run_swiss_roll.py) ----
    Y_init = build_svd_Y_init(D_asym, svd_k=args.svd_k, d=args.d,
                              normalize=not args.no_normalize_D)
    if not args.quiet:
        print(f"SVD Y_init: extent={Y_init.max()-Y_init.min():.2f}")

    # ---- B: virtual-point locate step, identical to run_swiss_roll.py ----
    B_located = locate_B(X, omega, k=args.k, emb_k=args.emb_k, neg=args.neg,
                         locate_epochs=args.locate_epochs, clip_delta=args.clip_delta,
                         seed=args.seed, verbose=not args.quiet)

    # ---- force-directed update: same mechanism as run_swiss_roll.py /
    # run_swiss_roll_isumap.py, just SVD-initialised Y instead of located Y ----
    out = randers_umap_fit(D_asym, n_neighbors=args.emb_k, n_negative_samples=args.neg,
                            n_epochs=args.epochs, use_drift=True,
                            B_fixed=B_located, Y_init_override=Y_init,
                            use_gravity=args.gravity, ramp=args.ramp,
                            clip_delta=args.clip_delta, snapshot_every=args.snapshot_every,
                            seed=args.seed, verbose=not args.quiet)
    Y, B = out["Y"], out["B"]
    limit = 1.0 - args.clip_delta

    if not args.quiet:
        bnorm = np.linalg.norm(B, axis=1)
        print(f"\nextent={Y.max()-Y.min():.2f}  mean||b||={bnorm.mean():.4f}  "
              f"max||b||={bnorm.max():.4f}  clipped={(bnorm >= limit - 1e-9).sum()}/{args.n}")

    # ---- plot --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 8))
    sc = ax.scatter(Y[:, 0], Y[:, 1], c=t, cmap="viridis", s=10, alpha=0.85, linewidths=0)
    plt.colorbar(sc, ax=ax, label="t (intrinsic coordinate)")

    bnorm = np.linalg.norm(B, axis=1)
    big = np.argsort(bnorm)[::-1][:25]
    if bnorm.max() > 0:
        sc_scale = 0.12 * (Y.max() - Y.min()) / bnorm.max()
        ax.quiver(Y[big, 0], Y[big, 1], B[big, 0] * sc_scale, B[big, 1] * sc_scale,
                  color="k", alpha=0.6, width=0.004, scale=1, scale_units="xy")

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"SVD Y-init + located-B + force-directed update  (n={args.n})", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=150)

    np.savez(f"{args.out}.npz", Y=Y, B=B, t=t, X=X, omega=omega)

    if not args.quiet:
        print(f"\nwrote {args.out}.png and {args.out}.npz")

    # ---- snapshot grid: SVD Y_init -> every N epochs -> final, side by side ---
    if args.snapshot_every is not None:
        snaps = out["snapshots"]
        n_snap = len(snaps)
        ncols = min(n_snap, 6)
        nrows = int(np.ceil(n_snap / ncols))
        fig2, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows),
                                   squeeze=False)
        vmin, vmax = t.min(), t.max()
        sc2 = None
        for idx, snap in enumerate(snaps):
            ax2 = axes[idx // ncols][idx % ncols]
            Yi = snap["Y"]
            sc2 = ax2.scatter(Yi[:, 0], Yi[:, 1], c=t, cmap="viridis", s=6,
                              alpha=0.85, linewidths=0, vmin=vmin, vmax=vmax)
            ax2.set_title(f"epoch {snap['epoch']}", fontsize=9)
            ax2.set_xticks([]); ax2.set_yticks([])
        for idx in range(n_snap, nrows * ncols):
            axes[idx // ncols][idx % ncols].axis("off")
        fig2.suptitle(f"SVD Y-init + located-B swiss-roll trajectory  (n={args.n}, "
                      f"snapshot_every={args.snapshot_every})", fontsize=11)
        if sc2 is not None:
            fig2.colorbar(sc2, ax=axes, label="t (intrinsic coordinate)",
                          fraction=0.02, pad=0.01)
        fig2.savefig(f"{args.out}_snapshots.png", dpi=150, bbox_inches="tight")

        if not args.quiet:
            print(f"wrote {args.out}_snapshots.png ({n_snap} snapshots)")


if __name__ == "__main__":
    main()
