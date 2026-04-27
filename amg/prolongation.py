"""
prolongation.py - Prolongation (interpolation) matrix construction.

This module handles graph partitioning via METIS and the construction
of the tentative and smoothed prolongation operators used in AMG.
"""

import logging
import numpy as np
import scipy.sparse as sp
import pymetis

logger = logging.getLogger(__name__)


def build_prolongation(
    A: sp.csr_matrix,
    n_parts: int,
    method: str = "smoothed",
    omega: float = 2 / 3,
    smoothing_passes: int = 1,
    normalize_P: bool = True,
    clip_P_negatives: bool = False,
) -> sp.csr_matrix:
    """
    Build a prolongation matrix P via METIS graph partitioning.

    Parameters
    ----------
    A : sp.csr_matrix
        The fine-level system matrix (N x N).
    n_parts : int
        Number of coarse-level aggregates (partitions).
    method : str, optional
        Prolongation method: 'binary' or 'smoothed'. Default is 'smoothed'.
    omega : float, optional
        Damping factor for Jacobi smoothing. Default is 2/3.
    smoothing_passes : int, optional
        Number of Jacobi smoothing passes applied to tentative P. Default is 1.
    normalize_P : bool, optional
        If True, normalize rows of P to sum to 1. Default is True.
    clip_P_negatives : bool, optional
        If True, clip negative entries in P to 0 before normalization.
        Default is False.

    Returns
    -------
    sp.csr_matrix
        The prolongation matrix P of shape (N, n_parts).

    Raises
    ------
    ValueError
        If an unknown method is specified.
    """
    N = A.shape[0]
    is_symmetric = (A != A.T).nnz == 0

    logger.info("Partitioning graph into %d clusters using METIS...", n_parts)
    logger.info("Constructing Prolongation (P) using '%s' method...", method)

    # METIS requires a symmetric adjacency
    adj_mat = A if is_symmetric else A + A.T
    adjacency = pymetis.CSRAdjacency(adj_mat.indptr, adj_mat.indices)
    _, membership = pymetis.part_graph(n_parts, adjacency=adjacency)
    membership = np.array(membership)

    # Tentative P: one non-zero (= 1) per row
    row_indices = np.arange(N)
    P_tentative = sp.csr_matrix(
        (np.ones(N), (row_indices, membership)),
        shape=(N, n_parts),
    )

    if method == "binary":
        P_final = P_tentative
    elif method == "smoothed":
        P_final = _apply_smoothing(
            A, P_tentative, omega, smoothing_passes, normalize_P, clip_P_negatives
        )
    else:
        raise ValueError(
            f"Unknown prolongation method: '{method}'. Choose 'binary' or 'smoothed'."
        )

    _log_prolongation_stats(P_final, method)
    return P_final


def _apply_smoothing(
    A: sp.csr_matrix,
    P_tent: sp.csr_matrix,
    omega: float,
    smoothing_passes: int,
    normalize_P: bool,
    clip_P_negatives: bool,
) -> sp.csr_matrix:
    """
    Apply Jacobi smoothing to the tentative prolongation operator.

    The smoothed prolongation is computed as:
        P <- (I - omega * D^{-1} * A)^k * P_tent

    where k is the number of smoothing passes and D is the diagonal of A.

    Parameters
    ----------
    A : sp.csr_matrix
        Fine-level system matrix.
    P_tent : sp.csr_matrix
        Tentative (binary) prolongation matrix.
    omega : float
        Jacobi damping parameter.
    smoothing_passes : int
        Number of smoothing iterations.
    normalize_P : bool
        Whether to normalize rows after smoothing.
    clip_P_negatives : bool
        Whether to clip negative entries before normalization.

    Returns
    -------
    sp.csr_matrix
        Smoothed prolongation matrix.
    """
    D = A.diagonal()
    D_safe = np.where(np.abs(D) < 1e-15, 1.0, D)
    D_inv = sp.diags(1.0 / D_safe)

    I = sp.eye(A.shape[0])
    Smoother = I - omega * D_inv @ A

    P = P_tent.copy().tocsr()
    for _ in range(max(1, smoothing_passes)):
        P = (Smoother @ P).tocsr()

    if clip_P_negatives:
        P.data = np.maximum(P.data, 0.0)
        P.eliminate_zeros()

    # Normalization must always follow clipping to restore the
    # partition of unity property (rows summing to 1).
    if normalize_P or clip_P_negatives:
        row_sums = np.array(P.sum(axis=1)).flatten()
        zero_rows = np.abs(row_sums) < 1e-15
        if np.any(zero_rows):
            logger.warning(
                "%d rows have near-zero sum after smoothing/clipping.", np.sum(zero_rows)
            )
            row_sums[zero_rows] = 1.0
        row_scaling = sp.diags(1.0 / row_sums)
        P = (row_scaling @ P).tocsr()

    return P


def _log_prolongation_stats(P: sp.csr_matrix, method_name: str) -> None:
    """Log diagnostic statistics about the prolongation matrix."""
    row_sums = np.array(P.sum(axis=1)).flatten()
    max_dev = np.max(np.abs(row_sums - 1.0))
    all_sum_to_one = np.allclose(row_sums, 1.0, rtol=1e-10, atol=1e-10)
    row_nnz = np.diff(P.indptr)

    logger.info("P matrix verification (%s):", method_name)
    logger.info("  Shape        : %s", P.shape)
    logger.info("  Non-zeros    : %d", P.nnz)
    logger.info(
        "  Row nnz      : min=%d, max=%d, mean=%.3f",
        row_nnz.min(), row_nnz.max(), row_nnz.mean(),
    )
    logger.info(
        "  Row sums     : min=%.12f, max=%.12f", row_sums.min(), row_sums.max()
    )
    logger.info("  Max dev from 1.0 : %.4e", max_dev)
    logger.info("  Rows sum to 1?   : %s", all_sum_to_one)
    logger.info("  Rows with 1 nnz  : %d", np.sum(row_nnz == 1))
