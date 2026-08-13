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


def run_located_drift(X, omega, k=15, emb_k=20, neg=10, locate_epochs=500,
                      epochs=500, clip_delta=0.01, use_gravity=False,
                      snapshot_every=None, ramp=False, seed=0, verbose=True):
    """
    The full two-step pipeline, factored out so both main() (CLI/plotting)
    and test.py (diagnostics) call the exact same code -- no duplication.

    STEP 1 locate : embed n real + n virtual (x_i+omega_i) points together,
                    plain UMAP (b=0), read off B := Y_virtual - Y_real.
    STEP 2 apply  : re-embed the n real points on the true D_asym, with that
                    B frozen + attached (B_fixed), optionally +gravity.

    snapshot_every : [OURS 2026-08-11] int or None. If given, forwarded to
        the APPLY step's randers_umap_fit call only (the locate step's
        result isn't what we usually want to watch evolve) -- captures Y
        every snapshot_every epochs, from the initial Y_real0 through to
        the final embedding, for a "training trajectory" plot.

    Returns
    -------
    dict: Y, B, D_asym, Y_real0, Y_virtual0, B_located (pre-apply, for
          diagnostics that want to inspect the locate step in isolation),
          snapshots (list of {"epoch", "Y", "B"}, only if snapshot_every given)
    """
    n = X.shape[0]

    # ---- locate: embed n real + n virtual (x_i+omega_i) points together ----
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

    # ---- apply: real D_asym, B frozen + attached ----------------------------
    if verbose:
        print(f"\nApply: building asymmetric D_asym on the {n} real points...")
    D_asym, _ = compute_dist_matrix(X, n_neighbors=k, path_method="auto",
                                    randers_field=omega)
    out2 = randers_umap_fit(D_asym, n_neighbors=emb_k, n_negative_samples=neg,
                            n_epochs=epochs, use_drift=True,
                            B_fixed=B_located, Y_init_override=Y_real0,
                            use_gravity=use_gravity, ramp=ramp,
                            snapshot_every=snapshot_every,
                            clip_delta=clip_delta, seed=seed, verbose=verbose)
    Y, B = out2["Y"], out2["B"]

    if verbose:
        bn = np.linalg.norm(B, axis=1)
        print(f"\nextent={Y.max()-Y.min():.2f}  mean||b||={bn.mean():.4f}  "
              f"max||b||={bn.max():.4f}  clipped={(bn >= limit - 1e-9).sum()}/{n}")

    result = {"Y": Y, "B": B, "D_asym": D_asym,
              "Y_real0": Y_real0, "Y_virtual0": Y_virtual0, "B_located": B_located}
    if snapshot_every is not None:
        result["snapshots"] = out2["snapshots"]
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n",      type=int, default=1000)
    p.add_argument("--k",      type=int, default=20, help="k-NN for the geodesic backbone")
    p.add_argument("--emb-k",  type=int, default=20, help="n_neighbors for randers_umap_fit")
    p.add_argument("--neg",    type=int, default=10)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--locate-epochs", type=int, default=500)
    p.add_argument("--clip-delta", type=float, default=0.01)
    p.add_argument("--gravity", action="store_true",
                    help="add per-node gravity = b_i to the apply step (no extra scaling)")
    p.add_argument("--snapshot-every", type=int, default=None,
                    help="if given, also save <out>_snapshots.png: the apply-step "
                         "embedding every N epochs (from Y_real0 to final), side by side")
    p.add_argument("--ramp", action="store_true",
                    help="[OURS 2026-08-11] ramp B_fixed's magnitude 0->1 over the first "
                         "70%% of apply-step epochs instead of applying it at full strength "
                         "from epoch 0 (default). Recommended for large/noisy n where the "
                         "located B has extreme (clipped) entries for many nodes -- full "
                         "strength from epoch 0 can fling those nodes out immediately, "
                         "producing central-collapse-plus-outliers instead of a smooth unroll.")
    p.add_argument("--seed",   type=int, default=0)
    p.add_argument("--out",    default="swiss_embedding")
    p.add_argument("--quiet",  action="store_true")
    args = p.parse_args()

    if not args.quiet:
        print(f"Generating swiss roll: n={args.n}")
    X, omega, t = make_swiss_roll_randers(args.n, seed=42)
    n = args.n

    result = run_located_drift(X, omega, k=args.k, emb_k=args.emb_k, neg=args.neg,
                               locate_epochs=args.locate_epochs, epochs=args.epochs,
                               clip_delta=args.clip_delta, use_gravity=args.gravity,
                               snapshot_every=args.snapshot_every, ramp=args.ramp,
                               seed=args.seed, verbose=not args.quiet)
    Y, B = result["Y"], result["B"]

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

    # ---- snapshot grid: init -> every N epochs -> final, side by side -------
    if args.snapshot_every is not None:
        snaps = result["snapshots"]
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
        fig2.suptitle(f"Randers-UMAP swiss-roll, apply-step trajectory  (n={n}, "
                      f"snapshot_every={args.snapshot_every})", fontsize=11)
        if sc2 is not None:
            fig2.colorbar(sc2, ax=axes, label="t (intrinsic coordinate)",
                          fraction=0.02, pad=0.01)
        fig2.savefig(f"{args.out}_snapshots.png", dpi=150, bbox_inches="tight")

        if not args.quiet:
            print(f"wrote {args.out}_snapshots.png ({n_snap} snapshots)")


if __name__ == "__main__":
    main()
