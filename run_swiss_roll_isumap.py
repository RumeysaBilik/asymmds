#!/usr/bin/env python3
"""
run_swiss_roll_isumap.py -- swiss roll, but the distance matrix comes from
distance_graph_generation() (the isumap method used in asymm_dist_MNIST.py),
NOT from randers_bridge.compute_dist_matrix().

Deliberately pulls isumap_dist[0] (data_D, the raw pre-t-conorm/pre-Dijkstra
neighbourhood distances from comp_graph()) exactly like asymm_dist_MNIST.py
does -- NOT the fully-processed D. This is intentional (per user), not the
"bug" discussed earlier: applying the t-conorm graph-merge symmetrises the
result, which is exactly what should be avoided here. distance_graph_generation.py
itself is untouched.

isumap's own D_asym construction (distance_graph_generation) has no notion
of the swiss roll's ground-truth Randers field (omega) -- it derives
asymmetry purely from the directed k-NN/star-graph/t-conorm structure of X,
not from an injected vector field. That part is unchanged and is the whole
point of comparison: how does isumap's own asymmetric graph behave under
the shared force-directed update.

BUT this is a synthetic validation dataset, so the true omega field IS available to us
even though isumap's own algorithm never looks at it. There's no reason to withhold it
just because isumap's D_asym construction doesn't use it -- B's source is a
separate, decoupled question from how D_asym is built. B is therefore
computed with the SAME virtual-point locate mechanism as run_swiss_roll.py
and run_swiss_roll_svd_radar.py:

    [OURS 2026-08-16, per explicit user request -- this replaces an older
    version of this paragraph that described "embed n real + n virtual
    points jointly with plain drift-off UMAP" i.e. a full locate_epochs-
    epoch force-directed training run before reading off B. That was
    replaced: training entangled B with SGD optimisation noise unrelated
    to the true x_i vs x_i+omega_i relationship. Current mechanism:]
    place n real + n virtual (x_i+omega_i) points with ONE deterministic
    spectral_layout call on their geodesic distance graph (no training,
    no iterations), then B_located := y_virtual - y_real (in that single
    placement), clipped to the Randers bound, frozen and passed as
    B_fixed. This makes the B SOURCE identical across all three
    pipelines -- all three now use spectral_layout-only, init-only locate.

This script no longer imports anything from
run_swiss_roll_svd_radar.py or svd_radar.py -- the SVD-of-Delta B variant
(build_fixed_B/--legacy-svd-b) is REMOVED entirely, not just unused, since
having it sit in the file (even gated behind a flag) was a recurring
source of confusion about what actually runs by default. The virtual-point
locate step is now self-contained below (locate_B_isumap()) instead of
importing locate_B() from run_swiss_roll_svd_radar.py, and it always uses
isumap's own build_isumap_dist_matrix() for the augmented-points geodesic
backbone (previously, even after B's source was unified across all three
scripts, this backbone still came from randers_bridge.compute_dist_matrix
via the imported locate_B -- that gap is closed here too, by construction,
since locate_B_isumap has no other distance function available to it).
Nothing in this file touches SVD anymore.

[OURS 2026-08-16, per explicit user request -- "temiz ve anlaşılır bir kod
istiyorum"] --live-drift (the old "recompute B every epoch" alternative to
located B) is REMOVED entirely, same reasoning as the SVD variant above:
an always-unused code path sitting behind a flag was clutter, not a real
alternative anyone was using. located B is now the only path.

t and alpha(t) (ground truth) are still known for swiss roll, so
test.py's direction_accuracy_swiss metric can still be run against this
output to see whether the isumap-derived asymmetry correlates with the
true field -- that comparison is the point of this script.

Usage
-----
    python3 run_swiss_roll_isumap.py
    python3 run_swiss_roll_isumap.py --n 2000 --epochs 500
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
from distance_graph_generation import distance_graph_generation
from randers_umap import randers_umap_fit, fuzzy_simplicial_set, spectral_layout


def build_isumap_dist_matrix(X, k=30, verbose=True):
    """Same recipe as asymm_dist_MNIST.py: pull data_D (isumap_dist[0]),
    NOT D -- reconstruct the (i,j,k) dict into a dense (n,n) matrix.

    [OURS 2026-08-07 bug fix] data_D only populates ~k entries per row
    (the epm=True "pure star graph" default -- see distance_graph_generation.py
    docstring). asymm_dist_MNIST.py's original reconstruction used
    np.zeros(), leaving every UN-populated (i,j) pair at exactly 0.0 --
    indistinguishable from a genuine zero distance. randers_umap.py's
    _knn_from_distance_matrix() picks the k SMALLEST values per row via
    argsort, so on a mostly-zero-filled row it was picking ~k phantom
    "distance-0" non-edges as the nearest neighbours instead of the ~k
    REAL populated ones (verified empirically: for k=20, all 20 selected
    neighbours were phantom zeros, 0/20 real). Filling the unpopulated
    entries with np.inf instead fixes this -- inf can never win an
    argsort-smallest selection, and downstream smooth_knn_dist/mu
    computations already handle inf gracefully (exp(-inf/sigma) = 0, so
    an inf "neighbour" that leaks into the top-k for an under-populated
    row just gets zero weight instead of corrupting the graph).
    """
    n = X.shape[0]
    isumap_dist = distance_graph_generation(
        X, k=k, normalize=True, distBeyondNN=True, verbose=verbose,
        dataIsDistMatrix=False, dataIsGeodesicDistMatrix=False, saveDistMatrix=False,
    )
    data_D = isumap_dist[0]
    D = np.full((n, n), np.inf)
    np.fill_diagonal(D, 0.0)
    for key, value in data_D.items():
        i, j, k_ = key
        D[i, j] = value
    return D


def locate_B_isumap(X, omega, k=15, emb_k=20, neg=10, locate_epochs=500,
                     clip_delta=0.01, seed=0, verbose=True):
    """
    [OURS 2026-08-15] Self-contained virtual-point locate step for B --
    same mechanism as run_swiss_roll.py's run_located_drift() STEP 1, but
    defined HERE instead of imported from run_swiss_roll_svd_radar.py -- per
    explicit user request, this file no longer pulls anything from a
    file named after a different (SVD-based) method. Always uses
    build_isumap_dist_matrix() for the augmented points' geodesic
    backbone, so this step is isumap-native end to end, same as D_asym.

    [OURS 2026-08-16, per explicit user request] B is now read off a SINGLE
    deterministic spectral_layout call on the augmented graph, not a fully
    -trained (locate_epochs-epoch) force-directed embedding -- training the
    augmented set entangled B with SGD optimisation noise unrelated to the
    true x_i vs x_i+omega_i relationship. locate_epochs is now a no-op,
    kept only for call-site/CLI compat. See run_swiss_roll.py's module
    docstring for the full rationale.
    """
    n = X.shape[0]
    X_virtual = X + omega
    X_aug = np.vstack([X, X_virtual])

    if verbose:
        print(f"\nLocate: building isumap-native geodesic D on {2*n} augmented points...")
    D_sym_aug = build_isumap_dist_matrix(X_aug, k=k, verbose=False)

    if verbose:
        print("Locate: spectral_layout on the augmented graph (no training)...")
    mu_aug, _ = fuzzy_simplicial_set(D_sym_aug, emb_k)
    Y_aug0 = spectral_layout(mu_aug, d=2, seed=seed)
    Y_real0, Y_virtual0 = Y_aug0[:n], Y_aug0[n:]

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
    p.add_argument("--k", type=int, default=30, help="k for distance_graph_generation (isumap's own D_asym)")
    p.add_argument("--emb-k", type=int, default=20, help="n_neighbors for randers_umap_fit (apply step)")
    p.add_argument("--neg", type=int, default=10)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--gravity", action="store_true")
    p.add_argument("--locate-k", type=int, default=15,
                    help="k-NN for the locate step's Isomap-style geodesic backbone "
                         "(compute_dist_matrix on the augmented real+virtual points) -- "
                         "separate from --k, which is isumap's own D_asym parameter")
    p.add_argument("--locate-epochs", type=int, default=500,
                    help="epochs for the virtual-point locate step (B)")
    p.add_argument("--clip-delta", type=float, default=0.01)
    p.add_argument("--ramp", action="store_true",
                    help="[OURS 2026-08-12] ramp B's magnitude 0->1 over the first 70%% of "
                         "epochs instead of applying it at full strength from epoch 0 "
                         "(default: off, matching run_swiss_roll.py -- B is located/computed "
                         "once and attached at full strength from the start).")
    p.add_argument("--init-only", action="store_true",
                    help="[OURS 2026-08-13] stop before force-directed training -- isumap has no "
                         "explicit separate Y_init step (randers_umap_fit computes its own spectral "
                         "init on D_asym internally), so this runs a single epoch with an internal "
                         "epoch-0 snapshot and returns that pre-training state instead of out['Y']/"
                         "out['B']. Ignores --epochs, --ramp, --gravity.")
    p.add_argument("--snapshot-every", type=int, default=None,
                    help="[OURS 2026-08-14] if given, also save <out>_snapshots.png: the apply-step "
                         "embedding (with drift-vector arrows) every N epochs, side by side. "
                         "Ignored if --init-only is also given.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="swiss_embedding_isumap")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    # [OURS 2026-08-13] init-only: run just 1 epoch with a snapshot_every=1 so
    # randers_umap_fit's guaranteed pre-loop epoch-0 capture gives us the raw
    # spectral Y_init + B (at ramp's epoch-0 value) without any real training.
    apply_epochs = 1 if args.init_only else args.epochs
    apply_snapshot_every = 1 if args.init_only else args.snapshot_every

    if not args.quiet:
        print(f"Generating swiss roll: n={args.n}")
    X, omega, t = make_swiss_roll_randers(args.n, seed=42)

    if not args.quiet:
        print(f"\nBuilding distance matrix via distance_graph_generation (data_D, unfixed)...")
    D_asym = build_isumap_dist_matrix(X, k=args.k, verbose=not args.quiet)

    # [OURS 2026-08-07] data_D only populates ~k-1 real entries per row (the
    # rest are the np.inf fill from the bug fix above). If emb_k requests
    # more neighbours than a row actually has, _knn_from_distance_matrix's
    # argsort is forced to include a phantom inf-distance "neighbour" to
    # fill the quota -- and N=0.5*(D_asym-D_asym.T) can be inf/nan exactly
    # there. Capping emb_k to the worst-case row's real neighbour count
    # guarantees every selected neighbour is real.
    min_real_neighbors = int(np.isfinite(D_asym).sum(axis=1).min() - 1)  # -1 excludes the diagonal
    emb_k = min(args.emb_k, max(min_real_neighbors, 1))
    if not args.quiet:
        print(f"D_asym: {D_asym.shape}  symmetric={np.allclose(D_asym, D_asym.T)}  "
              f"min real neighbours/row={min_real_neighbors}  emb_k used={emb_k}")

    # [OURS 2026-08-13, default; self-contained since 2026-08-15] located
    # B: SAME virtual-point mechanism as run_swiss_roll.py /
    # run_swiss_roll_svd_radar.py, using the true omega field --
    # available to us since this is a synthetic validation dataset,
    # even though isumap's own D_asym construction never touches
    # omega. locate_B_isumap() (defined above, in this file) builds
    # its own geodesic backbone on the augmented real+virtual points
    # via --locate-k, independent of isumap's --k, and always through
    # build_isumap_dist_matrix -- isumap-native end to end, no import
    # from run_swiss_roll_svd_radar.py.
    if not args.quiet:
        print(f"\nLocating B via virtual points (true omega, locate-epochs={args.locate_epochs}, "
              f"isumap-native locate backbone)...")
    B_fixed = locate_B_isumap(X, omega, k=args.locate_k, emb_k=args.emb_k, neg=args.neg,
                               locate_epochs=args.locate_epochs, clip_delta=args.clip_delta,
                               seed=args.seed, verbose=not args.quiet)
    out = randers_umap_fit(D_asym, n_neighbors=emb_k, n_negative_samples=args.neg,
                            n_epochs=apply_epochs, use_drift=True, B_fixed=B_fixed,
                            clip_delta=args.clip_delta,
                            use_gravity=args.gravity, ramp=args.ramp, seed=args.seed,
                            snapshot_every=apply_snapshot_every, verbose=not args.quiet)

    if args.init_only:
        # true pre-training state, captured before any epoch update
        Y, B = out["snapshots"][0]["Y"], out["snapshots"][0]["B"]
    else:
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
    drift_label = "located B (true omega)"
    init_suffix = ", INIT ONLY (no training)" if args.init_only else ""
    ax.set_title(f"Randers-UMAP, isumap-derived D, {drift_label}{init_suffix}  (n={args.n})", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=150)

    np.savez(f"{args.out}.npz", Y=Y, B=B, t=t, X=X, omega=omega)

    if not args.quiet:
        print(f"\nwrote {args.out}.png and {args.out}.npz")

    # ---- snapshot grid: init -> every N epochs -> final, side by side,
    # each panel with its own drift-vector arrows [OURS 2026-08-14] --------
    if args.snapshot_every is not None and not args.init_only:
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
        fig2.suptitle(f"Randers-UMAP, isumap-derived D, {drift_label}, trajectory  "
                      f"(n={args.n}, snapshot_every={args.snapshot_every})", fontsize=11)
        if sc2 is not None:
            fig2.colorbar(sc2, ax=axes, label="t (intrinsic coordinate)",
                          fraction=0.02, pad=0.01)
        fig2.savefig(f"{args.out}_snapshots.png", dpi=150, bbox_inches="tight")

        if not args.quiet:
            print(f"wrote {args.out}_snapshots.png ({n_snap} snapshots)")


if __name__ == "__main__":
    main()
