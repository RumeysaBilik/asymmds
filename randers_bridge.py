"""
randers_bridge.py
==================
Converts a raw point cloud X (n, m) plus a per-point Randers drift field
omega (n, m) into a full (n, n) asymmetric geodesic distance matrix D_asym,
suitable as direct input to randers_umap_fit() -- exactly the role that
load_migration_graph()'s D_asym plays for the migration dataset.

[OURS 2026-08-11] Adjacency construction rewritten to match
github.com/lwileczek/isomap's make_adjacency() (README "Step 1 Adjacency &
Distance Matrices", threshold variant) instead of sklearn's kneighbors_graph:

    dist = cdist(X, X)                     # full (n,n) pairwise distance
    adj  = inf everywhere
    adj[dist < eps] = dist[dist < eps]      # threshold, not k-NN membership
    D    = shortest_path(adj)

i.e. two points are an edge iff their Euclidean distance is below a
threshold `eps`, not iff one is among the other's k nearest neighbours.
Function name/signature kept as compute_dist_matrix() and n_neighbors kept
as the public knob (every caller in this repo passes n_neighbors=k) -- eps
is auto-derived from n_neighbors so nothing downstream has to change: eps
is set to the smallest radius such that every point has >= n_neighbors
neighbours within it (max, over all i, of point i's n_neighbors-th nearest
distance). Pass eps explicitly to bypass that and use lwileczek's raw
threshold knob directly.

Construction
------------
    1. Full pairwise Euclidean distance (scipy cdist), thresholded at eps
       -> adjacency matrix (lwileczek/isomap's method, not sklearn's k-NN
       graph).
    2. For every surviving edge (i, j):
           d(i, j) <- d(i, j) + <omega_i, x_j - x_i>
       (Randers-perturbed edge weight -- the discrete version of the
       continuous Randers metric F(x, v) = ||v|| + <omega_x, v>. This step
       has no counterpart in lwileczek/isomap -- it's the Finsler/Randers
       extension specific to this project.)
    3. Directed shortest-path (Dijkstra, via scipy's shortest_path -- the
       modern equivalent of the old sklearn.utils.graph_shortest_path that
       lwileczek/isomap's own code calls) over the resulting asymmetric
       weighted graph:
           D_asym[i, j] = geodesic distance i -> j,   in general != D_asym[j, i]
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.spatial.distance import cdist


def compute_dist_matrix(
        X,
        n_neighbors=5,
        eps=None,
        path_method="auto",
        metric="euclidean",
        randers_field=None,
        directed=None,
):
    """
    Parameters
    ----------
    X             : (n, m) raw coordinates
    n_neighbors   : used only to auto-derive eps when eps=None (see module
                    docstring) -- kept for call-site compatibility with the
                    rest of this repo, which all pass n_neighbors=k. This
                    is independent of randers_umap_fit's own n_neighbors,
                    which builds a *second*, UMAP-style fuzzy graph on top
                    of the D_asym this function returns.
    eps           : [OURS] lwileczek/isomap's actual threshold knob -- two
                    points are connected iff their Euclidean distance is
                    < eps. None (default) auto-derives eps from n_neighbors
                    so every point ends up with >= n_neighbors neighbours.
                    Pass a float to control the threshold directly instead.
    randers_field : (n, m) per-point drift vector omega_i (e.g. the `omega`
                    array from generated_swiss_roll-2.py), or None for the
                    plain Isomap-style geodesic distance
    directed      : bool or None. None (default): directed shortest-path
                    iff randers_field is given, undirected (symmetric
                    result) otherwise -- same semantics as before. Note the
                    eps-threshold adjacency (unlike sklearn's k-NN graph)
                    is symmetric by construction (dist(i,j)==dist(j,i)), so
                    with directed=True and randers_field=None there is no
                    longer any k-NN-membership asymmetry to isolate -- the
                    graph is symmetric until the Randers step perturbs it.
    path_method   : passed to scipy.sparse.csgraph.shortest_path

    Returns
    -------
    dist_matrix_ : (n, n) dense ndarray -- this is D_asym
    preds_       : (n, n) shortest-path predecessor matrix
    """
    n = X.shape[0]
    dist = cdist(X, X, metric=metric)
    np.fill_diagonal(dist, np.inf)  # exclude self so eps auto-derivation below ignores it

    if eps is None:
        # smallest per-point n_neighbors-th nearest distance, maxed over
        # all points -- guarantees every node has >= n_neighbors neighbours
        # within the threshold (this is lwileczek/isomap's own suggestion:
        # "tune your threshold so that each node has some minimum number
        # of connections").
        kth = np.sort(dist, axis=1)[:, n_neighbors - 1]
        eps_ = kth.max()
    else:
        eps_ = eps

    def _sparse_from_threshold(eps_val):
        # bln[i, j] True iff dist(i, j) < eps_val -- lwileczek/isomap's
        # edge rule. Only True entries are stored (true sparsity, unlike
        # a dense inf-filled array), matching what kneighbors_graph gave
        # us before.
        bln_ = dist < eps_val
        rows, cols = np.nonzero(bln_)
        vals = dist[rows, cols]
        return csr_matrix((vals, (rows, cols)), shape=(n, n)), bln_

    nbg, bln = _sparse_from_threshold(eps_)

    # [OURS] connectivity safety net -- lwileczek/isomap's own code has no
    # fallback for a disconnected graph (shortest_path just leaves
    # unreachable pairs at inf). We widen eps until connected, since a
    # graph full of infs breaks every downstream step (SVD, Adam loss,
    # UMAP fuzzy graph) rather than just degrading gracefully.
    n_components, _ = connected_components(nbg)
    while n_components > 1:
        eps_ *= 1.5
        nbg, bln = _sparse_from_threshold(eps_)
        n_components, _ = connected_components(nbg)

    # ── the actual Randers injection (no counterpart in lwileczek/isomap) ──
    if randers_field is not None:
        rows, cols = np.nonzero(bln)
        randers_update = np.einsum("ij,ij->i", X[cols] - X[rows], randers_field[rows])
        vals = dist[rows, cols] + randers_update
        nbg = csr_matrix((vals, (rows, cols)), shape=(n, n))
        directed_ = True if directed is None else directed
    else:
        directed_ = False if directed is None else directed

    dist_matrix_, preds_ = shortest_path(nbg, method=path_method, directed=directed_,
                                          return_predecessors=True)

    if X.dtype == np.float32:
        dist_matrix_ = dist_matrix_.astype(X.dtype, copy=False)

    return dist_matrix_, preds_
