#!/usr/bin/env python3
"""
run_swiss_roll.py -- Randers-UMAP on the swiss-roll + Randers field dataset,
drift (B) initialised via the "locate the target, then attach" method.

    b_i^virtual target := x_i + omega_i                (real ambient point)
    place n real + n virtual points with ONE spectral_layout call (no SGD
        training on the augmented set -- see [OURS 2026-08-16] below)
    b_i := y_i^virtual - y_i                            (in that placement)
    embed the n real points on the true D_asym, with that b_i frozen +
        attached (B_fixed) -- only a_i (the real point's own position)
        trains each epoch, b_i never changes

    the locate step used to train the augmented real+virtual system for locate_epochs (default 500)
    iterations of full force-directed randers_umap_fit before reading off
    B := Y_virtual - Y_real. That entangles B with the stochastic
    optimisation dynamics (negative sampling, attraction/repulsion) of an
    essentially separate embedding problem -- not a faithful readout of the
    true geometric relationship between x_i and x_i+omega_i. Replaced with
    a single deterministic spectral_layout call on the augmented graph
    (the same method UMAP itself uses as ITS OWN initialisation, precisely
    because it captures global structure without needing SGD refinement).
    --locate-epochs is now a no-op, kept only for CLI/call-site compat.

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
from randers_umap import randers_umap_fit, fuzzy_simplicial_set, spectral_layout, classical_mds


def make_swiss_roll_randers(n, seed=42):
    """Exact construction from generated_swiss_roll-2.py, parameterised by n."""
    rng = np.random.RandomState(seed)

    t = 1.5 * np.pi * (1 + 2 * rng.rand(n))
    height = 21 * rng.rand(n)

    x = t * np.cos(t)
    z = t * np.sin(t)
    y = height
    X = np.column_stack([x, y, z])

    # [OURS, original] pure rotational/angular tangent -- a 90-degree
    # rotation of the (x,z) position, NOT the true derivative of the
    # spiral curve. Kept here (commented, not deleted) so both versions
    # can be toggled between -- this one differs from DAGES's own true
    # tangent field by roughly 4-12 degrees over our t-range (missing the
    # radial-growth component -- see conversation/report for the derivation).
    #V = np.column_stack([-X[:, 2], np.zeros(n), X[:, 0]])

    # [OURS 2026-08-16, per DAGES's actual main_swiss_roll_full.py] true
    # tangent-to-the-spiral direction: d/dt (t*cos t, t*sin t), the exact
    # derivative of the parametric curve -- includes the radial-growth
    # component the commented-out V above was missing. This is what makes
    # the ground-truth field match DAGES's own swiss-roll experiment.
    tangent_x = np.cos(t) - t * np.sin(t)
    tangent_z = np.sin(t) + t * np.cos(t)
    V = np.column_stack([tangent_x, np.zeros(n), tangent_z])
    V = V / np.linalg.norm(V, axis=1)[:, None]

    #alpha = -0.5 * np.cos(t) + 0.3 * np.sin(t)
    alpha = 0.5
    #omega = alpha[:, None] * V
    omega = alpha * V


    return X, omega, t


def run_located_drift(X, omega, k=15, emb_k=20, neg=10, locate_epochs=500,
                      epochs=500, clip_delta=0.01, use_gravity=False,
                      gravity_strength=1.0, gravity_neighbor_weight=True,
                      snapshot_every=None, ramp=False, seed=0, verbose=True,
                      apply_step=True, init_method="umap"):
    """
    The full two-step pipeline, factored out so both main() (CLI/plotting)
    and test.py (diagnostics) call the exact same code -- no duplication.

    STEP 1 locate : place n real + n virtual (x_i+omega_i) points with ONE
                    deterministic placement call (no training), read off
                    B := Y_virtual - Y_real from that placement.
    STEP 2 apply  : embed the n real points on the true D_asym, with that
                    B frozen + attached (B_fixed), optionally +gravity --
                    only a_i (each real point's own position) trains each
                    epoch, b_i is never touched.

    init_method : [OURS 2026-08-16, per explicit user request] "umap"
        (default, unchanged) uses fuzzy_simplicial_set+spectral_layout --
        UMAP's own Laplacian-eigenmap init. "isomap" uses classical_mds
        instead -- Isomap's own finishing step (D_sym_aug here is already
        Isomap-style k-NN+Dijkstra via compute_dist_matrix, fully dense, so
        this makes the WHOLE pipeline consistently Isomap, not just the
        distance construction). Only affects STEP 1's placement method;
        everything else (B extraction, apply-step training) is identical.

    snapshot_every : [OURS 2026-08-11] int or None. If given, forwarded to
        the APPLY step's randers_umap_fit call only (the locate step's
        result isn't what we usually want to watch evolve) -- captures Y
        every snapshot_every epochs, from the initial Y_real0 through to
        the final embedding, for a "training trajectory" plot.

    apply_step : [OURS 2026-08-13] bool, default True. If False, skip STEP 2
        entirely -- no D_asym build, no force-directed training -- and
        return with "Y"/"B" set to the raw locate-step output (Y_real0 /
        B_located). Lets callers get just the initial (located) embedding
        with its drift vectors, e.g. for a quick "--init-only" mode.

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

    # [OURS 2026-08-16] init-only: ONE deterministic placement call on the
    # augmented graph, no force-directed training. locate_epochs is
    # intentionally unused now -- kept as a parameter only so existing
    # callers/CLI flags don't break.
    if init_method == "isomap":
        if verbose:
            print(f"Locate: classical_mds on the augmented graph (no training)...")
        Y_aug0 = classical_mds(D_sym_aug, d=2, seed=seed)
    elif init_method == "umap":
        if verbose:
            print(f"Locate: spectral_layout on the augmented graph (no training)...")
        mu_aug, _ = fuzzy_simplicial_set(D_sym_aug, emb_k)
        Y_aug0 = spectral_layout(mu_aug, d=2, seed=seed)
    else:
        raise ValueError(f"init_method must be 'umap' or 'isomap', got {init_method!r}")
    Y_real0, Y_virtual0 = Y_aug0[:n], Y_aug0[n:]

    B_located = Y_virtual0 - Y_real0
    limit = 1.0 - clip_delta
    bn0 = np.linalg.norm(B_located, axis=1, keepdims=True)
    B_located = B_located * np.minimum(1.0, limit / np.maximum(bn0, 1e-12))

    if verbose:
        bn = np.linalg.norm(B_located, axis=1)
        print(f"B located: mean||b||={bn.mean():.4f}  max||b||={bn.max():.4f}  "
              f"clipped={(bn >= limit - 1e-9).sum()}/{n}")

    if not apply_step:
        # [OURS 2026-08-13] init-only: stop here, hand back the raw located
        # embedding/drift as "Y"/"B" so callers (main()'s plotting code) can
        # treat this exactly like a normal result -- no D_asym built, no
        # training run.
        if verbose:
            print("\napply_step=False -- skipping STEP 2, returning located init only.")
        return {"Y": Y_real0, "B": B_located, "D_asym": None,
                "Y_real0": Y_real0, "Y_virtual0": Y_virtual0, "B_located": B_located}

    # ---- apply: real D_asym, B frozen + attached ----------------------------
    if verbose:
        print(f"\nApply: building asymmetric D_asym on the {n} real points...")
    D_asym, _ = compute_dist_matrix(X, n_neighbors=k, path_method="auto",
                                    randers_field=omega)
    out2 = randers_umap_fit(D_asym, n_neighbors=emb_k, n_negative_samples=neg,
                            n_epochs=epochs, use_drift=True,
                            B_fixed=B_located, Y_init_override=Y_real0,
                            use_gravity=use_gravity, gravity_strength=gravity_strength,
                            gravity_neighbor_weight=gravity_neighbor_weight, ramp=ramp,
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
    p.add_argument("--locate-epochs", type=int, default=500,
                    help="[OURS 2026-08-16] no-op -- locate step is now a single "
                         "spectral_layout call, no training. Kept for compat.")
    p.add_argument("--clip-delta", type=float, default=0.01)
    p.add_argument("--gravity", action="store_true",
                    help="[OURS 2026-08-17] add per-node gravity toward xi_i=y_i+b_i "
                         "(Bannister et al. f_g=gamma*M[i]*b_i), weighted by "
                         "--gravity-neighbor-weight unless disabled.")
    p.add_argument("--gravity-strength", type=float, default=1.0,
                    help="[OURS 2026-08-17] gamma_t in Bannister et al.'s gravity force. "
                         "Only matters with --gravity.")
    p.add_argument("--no-gravity-neighbor-weight", action="store_true",
                    help="[OURS 2026-08-17] disable the neighbour-plausibility weighting "
                         "(revert to the old unconditional gravity pull). Only matters with "
                         "--gravity.")
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
    p.add_argument("--init-only", action="store_true",
                    help="[OURS 2026-08-13] stop after the locate step -- skip the "
                         "force-directed apply/training step entirely, and just plot/save "
                         "the raw located embedding (Y_real0) with its drift vectors "
                         "(B_located). Ignores --epochs, --ramp, --gravity, "
                         "--snapshot-every (there's no training loop to snapshot).")
    p.add_argument("--init-method", choices=["umap", "isomap"], default="umap",
                    help="[OURS 2026-08-16] locate step's placement method. 'umap' (default) "
                         "= fuzzy_simplicial_set+spectral_layout. 'isomap' = classical_mds "
                         "(Isomap's own finishing step) -- makes the whole pipeline "
                         "consistently Isomap-style, not just the distance construction.")
    p.add_argument("--seed",   type=int, default=0)
    p.add_argument("--out",    default="swiss_embedding")
    p.add_argument("--quiet",  action="store_true")
    args = p.parse_args()

    if not args.quiet:
        print(f"Generating swiss roll: n={args.n}")
    X, omega, t = make_swiss_roll_randers(args.n, seed=42)
    n = args.n

    # ---- 3D plot of the ambient swiss roll with the omega (Randers) field ---
    fig3d = plt.figure(figsize=(11, 9))
    ax3d = fig3d.add_subplot(111, projection="3d")
    sc3d = ax3d.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap="viridis", s=8,
                        alpha=0.85, linewidths=0)
    fig3d.colorbar(sc3d, ax=ax3d, label="t (intrinsic coordinate)", shrink=0.6, pad=0.08)
    rng3d = np.random.RandomState(0)
    idx3d = rng3d.choice(n, size=min(150, n), replace=False)
    scale3d = 4.0
    ax3d.quiver(X[idx3d, 0], X[idx3d, 1], X[idx3d, 2],
                omega[idx3d, 0] * scale3d, omega[idx3d, 1] * scale3d, omega[idx3d, 2] * scale3d,
                color="k", alpha=0.7, linewidth=1.0, arrow_length_ratio=0.3)
    ax3d.set_title(f"Swiss roll (ambient X, n={n}) with omega field", fontsize=11)
    ax3d.set_xlabel("x"); ax3d.set_ylabel("y (height)"); ax3d.set_zlabel("z")
    fig3d.tight_layout()
    fig3d.savefig(f"{args.out}_3d_field.png", dpi=150)
    if not args.quiet:
        print(f"wrote {args.out}_3d_field.png")

    result = run_located_drift(X, omega, k=args.k, emb_k=args.emb_k, neg=args.neg,
                               locate_epochs=args.locate_epochs, epochs=args.epochs,
                               clip_delta=args.clip_delta, use_gravity=args.gravity,
                               gravity_strength=args.gravity_strength,
                               gravity_neighbor_weight=not args.no_gravity_neighbor_weight,
                               snapshot_every=args.snapshot_every, ramp=args.ramp,
                               seed=args.seed, verbose=not args.quiet,
                               apply_step=not args.init_only, init_method=args.init_method)
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
    if args.init_only:
        ax.set_title(f"Randers-UMAP swiss-roll, LOCATED INIT ONLY ({args.init_method}, no training)  (n={n})", fontsize=11)
    else:
        ax.set_title(f"Randers-UMAP swiss-roll, located-drift init ({args.init_method})  (n={n})", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=150)

    np.savez(f"{args.out}.npz", Y=Y, B=B, t=t, X=X, omega=omega)

    if not args.quiet:
        print(f"\nwrote {args.out}.png and {args.out}.npz")

    # ---- snapshot grid: init -> every N epochs -> final, side by side -------
    if args.snapshot_every is not None and not args.init_only:
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
            Bi = snap["B"]
            sc2 = ax2.scatter(Yi[:, 0], Yi[:, 1], c=t, cmap="viridis", s=6,
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
