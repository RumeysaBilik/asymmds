"""
randers_bridge.py
==================
[Ported from the FinslerMDS repo's utils.py::compute_dist_matrix -- logic
unchanged, stripped to just this one routine so it has no dependency on the
rest of that repo (Isomap wrapper, plotting helpers, etc).]

Converts a raw point cloud X (n, m) plus a per-point Randers drift field
omega (n, m) into a full (n, n) asymmetric geodesic distance matrix D_asym,
suitable as direct input to randers_umap_fit() -- exactly the role that
load_migration_graph()'s D_asym plays for the migration dataset.

Construction
------------
    1. Euclidean k-NN graph on X (sklearn kneighbors_graph).
    2. For every graph edge (i, j):
           d(i, j) <- d(i, j) + <omega_i, x_j - x_i>
       (Randers-perturbed edge weight -- the discrete version of the
       continuous Randers metric F(x, v) = ||v|| + <omega_x, v>.)
    3. Directed shortest-path (Dijkstra, via scipy's shortest_path) over the
       resulting asymmetric weighted graph:
           D_asym[i, j] = geodesic distance i -> j,   in general != D_asym[j, i]
"""

import numpy as np
from scipy.sparse.csgraph import connected_components, shortest_path
from sklearn.neighbors import NearestNeighbors, kneighbors_graph, radius_neighbors_graph
from sklearn.utils.graph import _fix_connected_components


def compute_dist_matrix(
        X,
        n_neighbors=5,
        radius=None,
        path_method="auto",
        neighbors_algorithm="auto",
        n_jobs=None,
        metric="minkowski",
        p=2,
        metric_params=None,
        randers_field=None,
):
    """
    Parameters
    ----------
    X             : (n, m) raw coordinates
    n_neighbors   : k for the Euclidean k-NN graph that forms the geodesic
                    backbone (this is independent of randers_umap_fit's own
                    n_neighbors, which builds a *second*, UMAP-style fuzzy
                    graph on top of the D_asym this function returns)
    randers_field : (n, m) per-point drift vector omega_i (e.g. the `omega`
                    array from generated_swiss_roll-2.py), or None for the
                    plain symmetric Isomap-style geodesic distance
    path_method   : passed to scipy.sparse.csgraph.shortest_path

    Returns
    -------
    dist_matrix_ : (n, n) dense ndarray -- this is D_asym
    preds_       : (n, n) shortest-path predecessor matrix
    """
    nbrs_ = NearestNeighbors(
        n_neighbors=n_neighbors, radius=radius, algorithm=neighbors_algorithm,
        metric=metric, p=p, metric_params=metric_params, n_jobs=n_jobs,
    )
    nbrs_.fit(X)

    if n_neighbors is not None:
        nbg = kneighbors_graph(
            nbrs_, n_neighbors, metric=metric, p=p,
            metric_params=metric_params, mode="distance", n_jobs=n_jobs,
        )
    else:
        nbg = radius_neighbors_graph(
            nbrs_, radius=radius, metric=metric, p=p,
            metric_params=metric_params, mode="distance", n_jobs=n_jobs,
        )

    # Make sure the graph is connected (same fix Isomap itself applies)
    n_connected_components, labels = connected_components(nbg)
    if n_connected_components > 1:
        nbg = _fix_connected_components(
            X=nbrs_._fit_X, graph=nbg, n_connected_components=n_connected_components,
            component_labels=labels, mode="distance", metric=nbrs_.effective_metric_,
            **nbrs_.effective_metric_params_,
        )

    # ── the actual Randers injection ────────────────────────────────────────
    if randers_field is not None:
        edges_mask = nbg.toarray() != 0
        for i in range(len(X)):
            randers_update = np.dot(X - X[i], randers_field[i]) * edges_mask[i]
            nbg[i, edges_mask[i]] = nbg[i, edges_mask[i]] + randers_update[edges_mask[i]]
        nbg = nbg.tocsr()
        directed = True
    else:
        directed = False

    dist_matrix_, preds_ = shortest_path(nbg, method=path_method, directed=directed,
                                          return_predecessors=True)

    if nbrs_._fit_X.dtype == np.float32:
        dist_matrix_ = dist_matrix_.astype(nbrs_._fit_X.dtype, copy=False)

    return dist_matrix_, preds_
