"""
svd_radar.py
============
[Ported from asymmetricfinsler-mds_demo/finsler_mds.py::svd_init -- logic
unchanged, stripped out so fdg has no dependency on the rest of that file
(the Adam/SMACOF training loops, etc). See SVD_Init_Report for the full
derivation and the CMDS-vs-SVD comparison this was validated against.]

RADAR-style ("Definition 1 / Algorithm 1" in the report) asymmetry-aware
embedding via truncated SVD of the asymmetric distance matrix itself --
NO double-centering / Gram-matrix step like classical MDS (which requires
a symmetric input). Given D (n,n), truncated SVD gives D ~= U_k S_k V_k^T;
the outgoing embedding is X_L = U_k sqrt(S_k), incoming is X_R = V_k sqrt(S_k),
and X = [X_L | X_R] is (n, 2k) -- X_L @ X_R.T ~= D, so this single matrix is
"asymmetry-aware" in the sense that two different linear projections
(picking the X_L vs X_R half) bilinearly reconstruct the asymmetric D.
"""

import numpy as np
from sklearn.decomposition import TruncatedSVD


def svd_init(D, k=10, normalize=True):
    """
    SVD-based (RADAR-style) embedding for an asymmetric (n, n) matrix D.

    Args:
        D: Asymmetric distance/skew matrix (n x n)
        k: Number of singular values to keep
        normalize: Whether to z-score normalize D first

    Returns:
        X: (n, 2k) -- [outgoing signal | incoming signal]
    """
    if normalize:
        mu = np.mean(D)
        sigma = np.std(D)
        D = (D - mu) / (sigma + 1e-8)

    svd = TruncatedSVD(n_components=k, random_state=42)
    U = svd.fit_transform(D)      # already U_orthonormal * S
    V = svd.fit_transform(D.T)    # already V_orthonormal * S

    sqrt_S = np.sqrt(np.maximum(svd.singular_values_, 1e-12))
    X_L = U / sqrt_S   # outgoing (source) signal
    X_R = V / sqrt_S   # incoming (destination) signal
    X = np.hstack([X_L, X_R])
    return X
