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

[OURS 2026-08-13, per explicit user request] BUT this is a synthetic
validation dataset, so the true omega field IS available to us even though
isumap's own algorithm never looks at it. There's no reason to withhold it
just because isumap's D_asym construction doesn't use it -- B's source is a
separate, decoupled question from how D_asym is built. B is therefore now
computed with the SAME virtual-point locate mechanism as run_swiss_roll.py
and run_swiss_roll_svd_radar.py (locate_B(), imported from
run_swiss_roll_svd_radar.py -- no duplicated logic): embed n real + n
virtual (x_i+omega_i) points jointly with plain drift-off UMAP on their
symmetric geodesic distance, then B_located := y_virtual - y_real, clipped
to the Randers bound, frozen and passed as B_fixed. This makes the B
SOURCE identical across all three pipelines -- the only thing that still
varies between them is D_asym / Y_init (isumap's own graph here, vs.
compute_dist_matrix's Isomap-style backbone elsewhere).

[SUPERSEDED, kept for reference] Previously (2026-08-10) B was computed via
a one-shot SVD of D_asym's skew-symmetric part (build_fixed_B(), still
defined below but no longer called by default) -- deliberately omega-free,
consistent with isumap's own premise. That comparison (SVD-of-Delta vs.
located-B, both applied to the same isumap D_asym) is still a valid thing
to run via --legacy-svd-b if wanted. Before that (even earlier), B was live
-- recomputed every epoch from the frozen asymmetric graph via
compute_drift(), exactly like embed_MNIST.py -- still available via
--live-drift.

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
from run_swiss_roll_svd_radar import locate_B
from distance_graph_generation import distance_graph_generation
from randers_umap import randers_umap_fit
from svd_radar import svd_init


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


def build_fixed_B(D_asym, svd_k=10, d=2, clip_delta=0.01):
    """[OURS 2026-08-10] SVD-of-Delta fixed-B, same recipe as
    finsler_mds_joint.py's B_init -- computed once, frozen, used as
    randers_umap_fit's B_fixed. D_asym has np.inf fill for unpopulated
    (i,j) entries (see build_isumap_dist_matrix docstring) -- SVD can't
    handle inf, so those entries are treated as "no signal" (0) here,
    same convention randers_umap.py's own N=D_asym-D_asym.T computation
    already uses (np.where(np.isfinite(N), N, 0.0)).
    """
    D_finite = np.where(np.isfinite(D_asym), D_asym, 0.0)
    Delta = D_finite.T - D_finite
    B = svd_init(Delta, k=svd_k, normalize=True)[:, :d]
    limit = 1.0 - clip_delta
    bn = np.linalg.norm(B, axis=1, keepdims=True)
    B = B * np.where(bn > limit, limit / np.maximum(bn, 1e-12), 1.0)
    return B


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
    p.add_argument("--legacy-svd-b", action="store_true",
                    help="use the OLD (2026-08-10) omega-free B: one-shot SVD of D_asym's "
                         "skew-symmetric part, instead of the located B (default, uses true omega)")
    p.add_argument("--svd-k", type=int, default=10,
                    help="singular values kept for --legacy-svd-b's fixed-B construction (unused otherwise)")
    p.add_argument("--clip-delta", type=float, default=0.01)
    p.add_argument("--live-drift", action="store_true",
                    help="use the OLD live drift (recomputed every epoch) instead of fixed B")
    p.add_argument("--ramp", action="store_true",
                    help="[OURS 2026-08-12] ramp B's magnitude 0->1 over the first 70%% of "
                         "epochs instead of applying it at full strength from epoch 0 "
                         "(default: off, matching run_swiss_roll.py -- B is located/computed "
                         "once and attached at full strength from the start).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="swiss_embedding_isumap")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

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

    if args.live_drift:
        out = randers_umap_fit(D_asym, n_neighbors=emb_k, n_negative_samples=args.neg,
                                n_epochs=args.epochs, use_drift=True,
                                use_gravity=args.gravity, ramp=args.ramp, seed=args.seed,
                                verbose=not args.quiet)
    elif args.legacy_svd_b:
        # [SUPERSEDED 2026-08-10, kept via --legacy-svd-b] fixed B: computed
        # once via SVD of Delta, frozen for the whole run -- see
        # build_fixed_B docstring. Deliberately omega-free.
        B_fixed = build_fixed_B(D_asym, svd_k=args.svd_k, d=2, clip_delta=args.clip_delta)
        if not args.quiet:
            bn = np.linalg.norm(B_fixed, axis=1)
            print(f"B_fixed (SVD-of-Delta): mean||b||={bn.mean():.4f}  max||b||={bn.max():.4f}")
        out = randers_umap_fit(D_asym, n_neighbors=emb_k, n_negative_samples=args.neg,
                                n_epochs=args.epochs, use_drift=True, B_fixed=B_fixed,
                                clip_delta=args.clip_delta,
                                use_gravity=args.gravity, ramp=args.ramp, seed=args.seed,
                                verbose=not args.quiet)
    else:
        # [OURS 2026-08-13, default] located B: SAME virtual-point mechanism
        # as run_swiss_roll.py / run_swiss_roll_svd_radar.py, using the true
        # omega field -- available to us since this is a synthetic
        # validation dataset, even though isumap's own D_asym construction
        # never touches omega. Only D_asym/Y_init still comes from isumap's
        # own machinery; B's source is now identical across all three
        # pipelines. locate_B() builds its own (Isomap-style, symmetric,
        # drift-off) geodesic backbone on the augmented real+virtual points
        # via --locate-k, independent of isumap's --k.
        if not args.quiet:
            print(f"\nLocating B via virtual points (true omega, locate-epochs={args.locate_epochs})...")
        B_fixed = locate_B(X, omega, k=args.locate_k, emb_k=args.emb_k, neg=args.neg,
                            locate_epochs=args.locate_epochs, clip_delta=args.clip_delta,
                            seed=args.seed, verbose=not args.quiet)
        out = randers_umap_fit(D_asym, n_neighbors=emb_k, n_negative_samples=args.neg,
                                n_epochs=args.epochs, use_drift=True, B_fixed=B_fixed,
                                clip_delta=args.clip_delta,
                                use_gravity=args.gravity, ramp=args.ramp, seed=args.seed,
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
    if args.live_drift:
        drift_label = "live drift"
    elif args.legacy_svd_b:
        drift_label = "fixed B (SVD-of-Delta)"
    else:
        drift_label = "located B (true omega)"
    ax.set_title(f"Randers-UMAP, isumap-derived D, {drift_label}  (n={args.n})", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=150)

    np.savez(f"{args.out}.npz", Y=Y, B=B, t=t, X=X, omega=omega)

    if not args.quiet:
        print(f"\nwrote {args.out}.png and {args.out}.npz")


if __name__ == "__main__":
    main()
