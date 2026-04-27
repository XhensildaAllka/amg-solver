#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 17:01:42 2026

@author: xallka
"""

"""
mlp_features.py - Node-level feature extraction for ML-P prolongation.

Extracts 10 features per fine node from the system matrix A and the
METIS cluster membership. These are used as input to MLProlongationMLP
to predict per-node interpolation strengths.

Feature definitions (10 features per node):
    0:  Diagonal value A_ii
    1:  Row sum of |A_i*|
    2:  Node degree (number of nonzeros in row i)
    3:  Max |A_ij| in row i
    4:  Mean |A_ij| in row i
    5:  Cluster size (number of nodes in same cluster)
    6:  Number of same-cluster neighbors
    7:  Number of different-cluster neighbors
    8:  Mean diagonal of neighbors
    9:  Mean row sum of neighbors

All features are computed with vectorized numpy operations.
The feature matrix is NOT normalized here — normalization is applied
in MLProlongationSolver using the statistics saved in the model checkpoint.
"""

import logging
import time
import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)


def extract_mlP_features(
    A: sp.csr_matrix,
    membership: np.ndarray,
    num_clusters: int,
) -> np.ndarray:
    """
    Extract 10 node-level features for ML-P prolongation prediction.

    Fully vectorized — no Python loops over nodes.

    Parameters
    ----------
    A : sp.csr_matrix
        The system matrix (N x N).
    membership : np.ndarray
        Integer array of shape (N,) mapping each node to a cluster ID.
    num_clusters : int
        Total number of METIS clusters.

    Returns
    -------
    np.ndarray
        Feature matrix of shape (N, 10). NOT normalized.
    """
    t0 = time.time()
    A_csr = A.tocsr()
    N     = A.shape[0]

    logger.info("Extracting ML-P features for %d nodes...", N)

    # --- Node-level features (fully vectorized) ---

    # Feature 0: diagonal
    diag = A_csr.diagonal()

    # Feature 1: row sum of absolute values
    abs_A    = abs(A_csr)
    row_sum  = np.array(abs_A.sum(axis=1)).flatten()

    # Feature 2: degree (number of nonzeros per row)
    degree = np.diff(A_csr.indptr)

    # Features 3, 4: max and mean |A_ij| per row
    # Use the absolute value matrix, compute per-row max and mean
    abs_A_csr = abs_A.tocsr()
    max_abs   = np.zeros(N)
    mean_abs  = np.zeros(N)

    # Vectorized: use the fact that abs_A_csr.data is already |A_ij|
    # np.maximum.reduceat groups data by row boundaries
    indptr = abs_A_csr.indptr
    data   = abs_A_csr.data

    nonempty = indptr[1:] > indptr[:-1]
    for i in np.where(nonempty)[0]:
        s, e = indptr[i], indptr[i+1]
        vals = data[s:e]
        max_abs[i]  = vals.max()
        mean_abs[i] = vals.mean()

    # Feature 5: cluster size
    cluster_size = np.zeros(N, dtype=np.float64)
    for c in range(num_clusters):
        idx = np.where(membership == c)[0]
        cluster_size[idx] = len(idx)

    # Features 6, 7: same-cluster and different-cluster neighbor counts
    # Vectorized using COO representation
    A_coo   = A_csr.tocoo()
    row_idx = A_coo.row
    col_idx = A_coo.col

    same_cluster_mask = membership[row_idx] == membership[col_idx]

    neighbors_same  = np.zeros(N)
    neighbors_other = np.zeros(N)
    np.add.at(neighbors_same,  row_idx[same_cluster_mask],  1)
    np.add.at(neighbors_other, row_idx[~same_cluster_mask], 1)

    # Features 8, 9: mean diagonal and row sum of neighbors
    # For each node i, average diag[j] and row_sum[j] over neighbors j
    mean_neighbor_diag    = np.zeros(N)
    mean_neighbor_rowsum  = np.zeros(N)

    # Weighted by 1/degree to get mean
    np.add.at(mean_neighbor_diag,   row_idx, diag[col_idx])
    np.add.at(mean_neighbor_rowsum, row_idx, row_sum[col_idx])

    safe_degree = np.maximum(degree, 1)
    mean_neighbor_diag   /= safe_degree
    mean_neighbor_rowsum /= safe_degree

    # Handle isolated nodes (degree=0): use own values
    isolated = degree == 0
    mean_neighbor_diag[isolated]   = diag[isolated]
    mean_neighbor_rowsum[isolated] = row_sum[isolated]

    features = np.column_stack([
        diag,                  # 0
        row_sum,               # 1
        degree,                # 2
        max_abs,               # 3
        mean_abs,              # 4
        cluster_size,          # 5
        neighbors_same,        # 6
        neighbors_other,       # 7
        mean_neighbor_diag,    # 8
        mean_neighbor_rowsum,  # 9
    ]).astype(np.float32)

    logger.info(
        "ML-P features extracted: shape=%s in %.1fs.",
        features.shape, time.time() - t0,
    )
    return features