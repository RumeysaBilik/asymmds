#!/usr/bin/env python3
"""
run_mammoth_isumap.py -- isumap-derived D_asym + live-derived drift, on the
mammoth point cloud. EXACT SAME pipeline as run_swiss_roll_isumap.py
(build_isumap_dist_matrix imported directly, not duplicated) -- only the
dataset changes, per explicit user request ("sadece data setini
değiştirelim").

See run_mammoth.py's module docstring for the mammoth Randers-field
rationale (hand-crafted global-axis field, DAGES river/sea style -- no
paper precedent exists for this dataset, neither IsUMap's nor DAGES's).

[OURS 2026-08-18, per explicit user/advisor feedback, then reconciled per a
second explicit user request -- see run_swiss_roll_isumap.py's module
docstring for the full rationale, especially locate_B_from_D_asym()'s
docstring] B is no longer located via virtual points that peeked at the
mammoth's true omega field. Instead, B originates ENTIRELY from D_asym's
own asymmetry (compute_drift on N = (D_asym-D_asym.T)/(D_asym+D_asym.T),
no omega anywhere in that formula), but -- unlike a first attempt at this
fix -- it is NOT recomputed every epoch. It is computed ONCE, on the
untrained Y_init (spectral_layout on D_asym's own fuzzy graph), then frozen
and attached to each node for the whole apply-step training, exactly like
the old virtual-point mechanism used to do (just sourced from D_asym's
asymmetry instead of from omega). omega is only used to build the
mammoth's ambient point cloud + drift field in make_mammoth_randers(); it
is never read again after that (D_asym here comes from
distance_graph_generation, which has no randers_field parameter at all --
its asymmetry is purely the directed k-NN/star-graph structure of X, not
an injected vector field).

Usage
-----
    python run_mammoth_isumap.py
    python run_mammoth_isumap.py --n 1500 --epochs 500
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

from run_mammoth import make_mammoth_randers
from run_swiss_roll_isumap import build_isumap_dist_matrix, locate_B_from_D_asym
from randers_umap import randers_umap_fit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--k", type=int, default=30, help="k for distance_graph_generation (isumap's own D_asym)")
    p.add_argument("--emb-k", type=int, default=20, help="n_neighbors for randers_umap_fit (apply step)")
    p.add_argument("--neg", type=int, default=10)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--gravity", action="store_true",
                    help="[OURS 2026-08-17] add per-node gravity toward xi_i=y_i+b_i "
                         "(Bannister et al. f_g=gamma*M[i]*b_i).")
    p.add_argument("--gravity-strength", type=float, default=1.0)
    p.add_argument("--no-gravity-neighbor-weight", action="store_true")
    p.add_argument("--clip-delta", type=float, default=0.01)
    p.add_argument("--ramp", action="store_true")
    p.add_argument("--init-only", action="store_true",
                    help="stop before force-directed training -- runs a single epoch with an "
                         "internal epoch-0 snapshot and returns that pre-training state.")
    p.add_argument("--snapshot-every", type=int, default=None)
    p.add_argument("--alpha", type=float, default=0.5, help="max ||omega|| for the mammoth drift field")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="mammoth_embedding_isumap")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    apply_epochs = 1 if args.init_only else args.epochs
    apply_snapshot_every = 1 if args.init_only else args.snapshot_every

    if not args.quiet:
        print(f"Loading mammoth.csv, subsampling to n={args.n}")
    X, omega, z = make_mammoth_randers(args.n, seed=42, alpha=args.alpha)

    if not args.quiet:
        print(f"\nBuilding distance matrix via distance_graph_generation (data_D, unfixed)...")
    D_asym = build_isumap_dist_matrix(X, k=args.k, verbose=not args.quiet)

    min_real_neighbors = int(np.isfinite(D_asym).sum(axis=1).min() - 1)
    emb_k = min(args.emb_k, max(min_real_neighbors, 1))
    if not args.quiet:
        print(f"D_asym: {D_asym.shape}  symmetric={np.allclose(D_asym, D_asym.T)}  "
              f"min real neighbours/row={min_real_neighbors}  emb_k used={emb_k}")

    # [OURS 2026-08-18, per explicit user request -- see
    # locate_B_from_D_asym's docstring in run_swiss_roll_isumap.py] B is
    # derived purely from D_asym's own asymmetry (no omega), computed ONCE
    # on the untrained Y_init and frozen -- not recomputed every epoch.
    if not args.quiet:
        print(f"\nLocating B from D_asym's own asymmetry (no omega, single frozen calculation)...")
    B_fixed = locate_B_from_D_asym(D_asym, emb_k, clip_delta=args.clip_delta,
                                    seed=args.seed, verbose=not args.quiet)
    out = randers_umap_fit(D_asym, n_neighbors=emb_k, n_negative_samples=args.neg,
                            n_epochs=apply_epochs, use_drift=True, B_fixed=B_fixed,
                            clip_delta=args.clip_delta,
                            use_gravity=args.gravity, gravity_strength=args.gravity_strength,
                            gravity_neighbor_weight=not args.no_gravity_neighbor_weight,
                            ramp=args.ramp, seed=args.seed,
                            snapshot_every=apply_snapshot_every, verbose=not args.quiet)

    if args.init_only:
        Y, B = out["snapshots"][0]["Y"], out["snapshots"][0]["B"]
    else:
        Y, B = out["Y"], out["B"]

    # ---- plot --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 8))
    sc = ax.scatter(Y[:, 0], Y[:, 1], c=z, cmap="viridis", s=10, alpha=0.85, linewidths=0)
    plt.colorbar(sc, ax=ax, label="z (tail<->head)")

    bn = np.linalg.norm(B, axis=1)
    big = np.argsort(bn)[::-1][:25]
    if bn.max() > 0:
        sc_scale = 0.12 * (Y.max() - Y.min()) / bn.max()
        ax.quiver(Y[big, 0], Y[big, 1], B[big, 0] * sc_scale, B[big, 1] * sc_scale,
                  color="k", alpha=0.6, width=0.004, scale=1, scale_units="xy")

    ax.set_xticks([]); ax.set_yticks([])
    drift_label = "located B (from D_asym asymmetry only, frozen)"
    init_suffix = ", INIT ONLY (no training)" if args.init_only else ""
    ax.set_title(f"Randers-UMAP mammoth, isumap-derived D, {drift_label}{init_suffix}  (n={args.n})", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=150)

    np.savez(f"{args.out}.npz", Y=Y, B=B, z=z, X=X, omega=omega)

    if not args.quiet:
        print(f"\nwrote {args.out}.png and {args.out}.npz")

    if args.snapshot_every is not None and not args.init_only:
        snaps = out["snapshots"]
        n_snap = len(snaps)
        ncols = min(n_snap, 6)
        nrows = int(np.ceil(n_snap / ncols))
        fig2, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows),
                                   squeeze=False)
        vmin, vmax = z.min(), z.max()
        sc2 = None
        for idx, snap in enumerate(snaps):
            ax2 = axes[idx // ncols][idx % ncols]
            Yi = snap["Y"]
            Bi = snap["B"]
            sc2 = ax2.scatter(Yi[:, 0], Yi[:, 1], c=z, cmap="viridis", s=6,
                              alpha=0.85, linewidths=0, vmin=vmin, vmax=vmax)
            bni = np.linalg.norm(Bi, axis=1)
            bigi = np.argsort(bni)[::-1][:25]
            if bni.max() > 0:
                sc_scale_i = 0.12 * (Yi.max() - Yi.min()) / bni.max()
                ax2.quiver(Yi[bigi, 0], Yi[bigi, 1],
                          Bi[bigi, 0] * sc_scale_i, Bi[bigi, 1] * sc_scale_i,
                          color="k", alpha=0.6, width=0.006, scale=1, scale_units="xy")
            ax2.set_title(f"epoch {snap['epoch']}", fontsize=9)
            ax2.set_xticks([]); ax2.set_yticks([])
        for idx in range(n_snap, nrows * ncols):
            axes[idx // ncols][idx % ncols].axis("off")
        fig2.suptitle(f"Randers-UMAP mammoth, isumap D, apply-step trajectory  (n={args.n}, "
                      f"snapshot_every={args.snapshot_every})", fontsize=11)
        if sc2 is not None:
            fig2.colorbar(sc2, ax=axes, label="z (tail<->head)",
                          fraction=0.02, pad=0.01)
        fig2.savefig(f"{args.out}_snapshots.png", dpi=150, bbox_inches="tight")

        if not args.quiet:
            print(f"wrote {args.out}_snapshots.png ({n_snap} snapshots)")


if __name__ == "__main__":
    main()
