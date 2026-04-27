"""
mlp_solver.py - AMG solver with ML-predicted prolongation operator (ML-P).

This module provides MLProlongationSolver, which uses a trained MLP to
predict a per-node "interpolation strength" s_i in (0,1). This strength
controls how aggressively each fine node interpolates from neighboring
clusters when building the prolongation matrix P.

The key idea (vs classical smoothed aggregation):
    Classical:  alpha = 0.67 (fixed for all nodes)
    ML-P:       alpha_i = 0.67 * s_i  (learned per node)

Two prolongation methods are supported:
    'smoothed': P_ij = -alpha_i * D_i^-1 * A_ij  for j in other clusters
                       1 - sum(off-cluster weights)  for j = own cluster
    'direct':   P_ij = s_i * |A_ij| / |A_ii|  if above threshold
                       (threshold adapts to s_i)

Both methods are fully vectorized — no Python loops over nodes.
For N=1.1M nodes this reduces P construction from ~hours to ~seconds.

Usage
-----
    from amg.mlp_solver import MLProlongationSolver

    solver = MLProlongationSolver(
        A,
        mlP_model_path='mlP_model_suitesparse.pth',
        num_clusters=1133,
        prolongation_method='smoothed',
    )
    result = solver.solve(b, method='lgmres', tol=1e-10)
"""

import logging
import time
from typing import Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, gmres, cg, bicgstab

from .smoothers import make_coarse_solver
from .result import SolverResult

logger = logging.getLogger(__name__)

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch not found. MLProlongationSolver will not work.")


class MLProlongationSolver:
    """
    Two-level AMG solver with ML-predicted prolongation matrix.

    The MLP predicts a per-node interpolation strength s_i in (0,1),
    which controls how much each fine node interpolates from neighboring
    clusters. Nodes with high s_i get richer interpolation (more
    off-cluster connections); nodes with low s_i stay close to pure
    injection.

    P construction is fully vectorized for efficiency on large matrices.

    Parameters
    ----------
    A : sp.spmatrix
        The system matrix (N x N). Must be square and sparse.
    mlP_model_path : str
        Path to the trained .pth model file (MLProlongationMLP).
    num_clusters : int, optional
        Number of METIS clusters. Default is max(2, N // 1000).
    prolongation_method : str, optional
        'smoothed' (Jacobi-style) or 'direct' (threshold-based).
        Default is 'smoothed'.
    smoother_steps_pre : int, optional
        Pre-smoothing sweeps per V-cycle. Default is 2.
    smoother_steps_post : int, optional
        Post-smoothing sweeps per V-cycle. Default is 2.
    smoother_weight : float, optional
        Jacobi damping weight for the smoother. Default is 0.7.
    coarse_solver : str, optional
        'direct' (LU) or 'cg'. Default is 'direct'.
    alpha_scale : float, optional
        Scaling factor for off-cluster interpolation weights.
        Corresponds to the 0.67 factor in classical smoothed aggregation.
        Default is 0.67.
    direct_threshold_base : float, optional
        Base threshold for the 'direct' method. Default is 0.15.
    """

    def __init__(
        self,
        A: sp.spmatrix,
        mlP_model_path: str,
        num_clusters: Optional[int] = None,
        prolongation_method: str = "smoothed",
        smoother_steps_pre: int = 2,
        smoother_steps_post: int = 2,
        smoother_weight: float = 0.7,
        coarse_solver: str = "direct",
        alpha_scale: float = 0.67,
        direct_threshold_base: float = 0.15,
    ) -> None:
        self._validate_inputs(A)
        self.A = A.tocsr()
        self.N = A.shape[0]
        self.smoother_steps_pre  = smoother_steps_pre
        self.smoother_steps_post = smoother_steps_post
        self.smoother_weight     = smoother_weight
        self.alpha_scale         = alpha_scale
        self.direct_threshold_base = direct_threshold_base

        self.n_parts = num_clusters if num_clusters else max(2, self.N // 1000)

        logger.info(
            "Initializing MLProlongationSolver: N=%d, n_parts=%d, "
            "method='%s', coarse_solver='%s'",
            self.N, self.n_parts, prolongation_method, coarse_solver,
        )

        # 1. METIS partitioning
        self.membership = self._run_metis()

        # 2. ML predictions
        node_strengths = self._predict_node_strengths(mlP_model_path)

        # 3. Build P (vectorized)
        t0 = time.time()
        if prolongation_method == "smoothed":
            self.P = self._build_P_smoothed(node_strengths)
        elif prolongation_method == "direct":
            self.P = self._build_P_direct(node_strengths)
        else:
            raise ValueError(
                f"Unknown prolongation_method: '{prolongation_method}'. "
                "Choose 'smoothed' or 'direct'."
            )
        logger.info(
            "P built in %.1fs: shape=%s, nnz=%d",
            time.time() - t0, self.P.shape, self.P.nnz,
        )

        # 4. Log P diagnostics
        self._log_P_diagnostics()

        # 5. Build coarse matrix and factorize
        self.A_c = (self.P.T @ self.A @ self.P).tocsr()
        logger.info(
            "Coarse matrix: shape=%s, nnz=%d", self.A_c.shape, self.A_c.nnz
        )
        self._coarse_solver = make_coarse_solver(self.A_c, method=coarse_solver)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        b: np.ndarray,
        method: str = "lgmres",
        tol: float = 1e-10,
        max_iter: int = 500,
    ) -> SolverResult:
        """
        Solve A x = b using the ML-P V-cycle as preconditioner.

        Parameters
        ----------
        b : np.ndarray
            Right-hand side vector of length N.
        method : str, optional
            Outer Krylov method: 'cg', 'gmres', 'bicgstab', 'lgmres'.
            Default is 'lgmres'.
        tol : float, optional
            Convergence tolerance. Default is 1e-10.
        max_iter : int, optional
            Maximum outer iterations. Default is 500.

        Returns
        -------
        SolverResult
        """
        self._validate_rhs(b)

        M_inv = LinearOperator(
            shape=(self.N, self.N),
            matvec=self.v_cycle,
            dtype=self.A.dtype,
        )

        residuals = []
        iteration_counter = [0]

        def _track(xk):
            rel_res = np.linalg.norm(b - self.A @ xk) / (
                np.linalg.norm(b) + 1e-300
            )
            residuals.append(rel_res)
            iteration_counter[0] += 1

        def _track_gmres(rk):
            residuals.append(float(rk))
            iteration_counter[0] += 1

        logger.info(
            "Starting ML-P %s solve (tol=%.1e, max_iter=%d)...",
            method.upper(), tol, max_iter,
        )
        start_time = time.time()

        method_lower = method.lower()
        if method_lower == "cg":
            x, info = cg(
                self.A, b, M=M_inv, atol=tol,
                callback=_track, maxiter=max_iter,
            )
        elif method_lower == "bicgstab":
            x, info = bicgstab(
                self.A, b, M=M_inv, atol=tol,
                callback=_track, maxiter=max_iter,
            )
        elif method_lower == "lgmres":
            from scipy.sparse.linalg import lgmres
            x, info = lgmres(
                self.A, b, M=M_inv, atol=tol,
                callback=_track, maxiter=max_iter,
                inner_m=30, outer_k=3,
            )
        else:
            x, info = gmres(
                self.A, b, M=M_inv, atol=tol,
                callback=_track_gmres,
                callback_type="legacy",
                maxiter=max_iter,
            )

        solve_time = time.time() - start_time
        converged  = info == 0

        result = SolverResult(
            x=x,
            converged=converged,
            iterations=len(residuals),
            residual_history=residuals,
            solve_time=solve_time,
            info=info,
            method=f"mlp_{method_lower}",
            tol=tol,
        )
        logger.info(result)
        return result

    def v_cycle(self, f: np.ndarray) -> np.ndarray:
        """One ML-P V-cycle (used as preconditioner action)."""
        u = np.zeros_like(f)
        u = self._jacobi(f, u, steps=self.smoother_steps_pre)

        r_c = self.P.T @ (f - self.A @ u)
        c_c = self._coarse_solver.solve(r_c)
        u  += self.P @ c_c

        u = self._jacobi(f, u, steps=self.smoother_steps_post)
        return u

    # ------------------------------------------------------------------
    # P construction (vectorized)
    # ------------------------------------------------------------------

    def _build_P_smoothed(self, node_strengths: np.ndarray) -> sp.csr_matrix:
        """
        Build smoothed prolongation P using ML-predicted strengths.

        Vectorized implementation — no Python loops over nodes.

        For each fine node i:
            alpha_i = alpha_scale * s_i
            P_i,own = 1 - sum_{j: nc != own} (-alpha_i * D_i^-1 * A_ij)
            P_i,nc  = -alpha_i * D_i^-1 * A_ij  for j in cluster nc != own
        """
        logger.info("Building smoothed ML-P (vectorized)...")
        A = self.A
        N = self.N
        membership = self.membership

        D_inv = 1.0 / (A.diagonal() + 1e-12)
        alpha = self.alpha_scale * node_strengths  # shape (N,)

        # All (i, j) pairs from A in COO form
        A_coo   = A.tocoo()
        row_idx = A_coo.row
        col_idx = A_coo.col
        vals    = A_coo.data

        # Cluster assignment for each j
        col_cluster = membership[col_idx]
        row_cluster = membership[row_idx]

        # Off-cluster mask: j belongs to a different cluster than i
        off_mask = col_cluster != row_cluster

        # Off-cluster entries: weight = -alpha_i * D_i^-1 * A_ij
        off_rows    = row_idx[off_mask]
        off_cols    = col_cluster[off_mask]   # target column in P = cluster of j
        off_weights = -alpha[off_rows] * D_inv[off_rows] * vals[off_mask]

        # Own-cluster entries: start with 1.0, subtract sum of off-cluster weights
        # (own_weight = 1 - sum_{off} weight, note weights are negative so we add)
        own_adjustment = np.zeros(N)
        np.add.at(own_adjustment, off_rows, off_weights)

        own_rows    = np.arange(N)
        own_cols    = membership
        own_weights = 1.0 - own_adjustment

        # Combine off-cluster and own-cluster entries
        all_rows = np.concatenate([off_rows, own_rows])
        all_cols = np.concatenate([off_cols, own_cols])
        all_data = np.concatenate([off_weights, own_weights])

        P_raw = sp.csr_matrix(
            (all_data, (all_rows, all_cols)),
            shape=(N, self.n_parts),
        )

        # Row normalization
        row_sums = np.array(P_raw.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        P = sp.diags(1.0 / row_sums) @ P_raw
        return P.tocsr()

    def _build_P_direct(self, node_strengths: np.ndarray) -> sp.csr_matrix:
        """
        Build direct ML-P using threshold-based off-cluster connections.

        Vectorized implementation — no Python loops over nodes.

        For each off-cluster neighbor j of node i:
            score_ij = s_i * |A_ij| / |A_ii|
            if score_ij > threshold_i: add to P
        threshold_i = base * (1 - s_i * 0.5)   (adapts to node strength)
        """
        logger.info("Building direct ML-P (vectorized)...")
        A = self.A
        N = self.N
        membership = self.membership

        abs_diag = np.abs(A.diagonal()) + 1e-12

        A_coo   = A.tocoo()
        row_idx = A_coo.row
        col_idx = A_coo.col
        abs_vals = np.abs(A_coo.data)

        col_cluster = membership[col_idx]
        row_cluster = membership[row_idx]

        # Off-cluster, non-diagonal entries only
        off_mask = (col_cluster != row_cluster) & (row_idx != col_idx)
        off_rows  = row_idx[off_mask]
        off_cols  = col_cluster[off_mask]
        off_avals = abs_vals[off_mask]

        # Score and threshold
        scores     = node_strengths[off_rows] * off_avals / abs_diag[off_rows]
        thresholds = self.direct_threshold_base * (
            1.0 - node_strengths[off_rows] * 0.5
        )
        keep = scores > thresholds

        off_rows  = off_rows[keep]
        off_cols  = off_cols[keep]
        off_data  = scores[keep]

        # Own-cluster entries
        own_rows = np.arange(N)
        own_cols = membership
        own_data = np.ones(N)

        all_rows = np.concatenate([off_rows, own_rows])
        all_cols = np.concatenate([off_cols, own_cols])
        all_data = np.concatenate([off_data, own_data])

        P_raw = sp.csr_matrix(
            (all_data, (all_rows, all_cols)),
            shape=(N, self.n_parts),
        )

        row_sums = np.array(P_raw.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        P = sp.diags(1.0 / row_sums) @ P_raw
        return P.tocsr()

    # ------------------------------------------------------------------
    # ML inference
    # ------------------------------------------------------------------

    def _predict_node_strengths(self, mlP_model_path: str) -> np.ndarray:
        """
        Load model and predict per-node interpolation strengths.

        Returns array of shape (N,) with values in (0,1).
        Falls back to 0.5 (uniform) if model cannot be loaded.
        """
        fallback = np.full(self.N, 0.5)

        if not _TORCH_AVAILABLE:
            logger.warning("PyTorch not available. Using uniform strength=0.5.")
            return fallback

        try:
            from .mlp_model import load_mlP_model
            from .mlp_features import extract_mlP_features

            logger.info("Loading ML-P model from %s...", mlP_model_path)
            model, norm = load_mlP_model(mlP_model_path)

            feats = extract_mlP_features(
                self.A, self.membership, self.n_parts
            )
            feats_norm = (feats - norm["X_mean"]) / (norm["X_std"] + 1e-8)

            with torch.no_grad():
                raw = (
                    model(torch.tensor(feats_norm, dtype=torch.float32))
                    .numpy()
                    .flatten()
                )

            # Sigmoid to map to (0, 1)
            node_strengths = 1.0 / (1.0 + np.exp(-raw))

            logger.info(
                "ML-P strengths: min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
                node_strengths.min(), node_strengths.max(),
                node_strengths.mean(), node_strengths.std(),
            )
            return node_strengths

        except FileNotFoundError:
            logger.warning(
                "Model file '%s' not found. Using uniform strength=0.5.",
                mlP_model_path,
            )
            return fallback
        except Exception as e:
            logger.warning(
                "ML-P model loading failed (%s). Using uniform strength=0.5.",
                str(e),
            )
            return fallback

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_metis(self) -> np.ndarray:
        """Run METIS partitioning and return membership array."""
        import pymetis
        is_symmetric = (self.A != self.A.T).nnz == 0
        adj = self.A if is_symmetric else (self.A + self.A.T)
        logger.info(
            "Partitioning graph into %d clusters (METIS)...", self.n_parts
        )
        adjacency = pymetis.CSRAdjacency(adj.indptr, adj.indices)
        _, membership_list = pymetis.part_graph(
            self.n_parts, adjacency=adjacency
        )
        return np.array(membership_list)

    def _jacobi(
        self, f: np.ndarray, u: np.ndarray, steps: int
    ) -> np.ndarray:
        """Weighted Jacobi smoother (scalar weight)."""
        D = self.A.diagonal()
        with np.errstate(divide="ignore", invalid="ignore"):
            D_inv = np.where(np.abs(D) > 1e-15, 1.0 / D, 0.0)
        for _ in range(steps):
            u = u + self.smoother_weight * D_inv * (f - self.A @ u)
        return u

    def _log_P_diagnostics(self) -> None:
        """Log key statistics about the prolongation matrix."""
        P_csr = self.P.tocsr()
        nnz_per_row = np.diff(P_csr.indptr)
        n_injection = (nnz_per_row == 1).sum()
        n_interp    = (nnz_per_row >= 2).sum()

        logger.info("P diagnostics:")
        logger.info(
            "  Pure injection (1 nnz): %d / %d (%.1f%%)",
            n_injection, self.N, 100 * n_injection / self.N,
        )
        logger.info(
            "  Interpolation (2+ nnz): %d / %d (%.1f%%)",
            n_interp, self.N, 100 * n_interp / self.N,
        )
        logger.info(
            "  Row nnz: min=%d, max=%d, mean=%.2f",
            nnz_per_row.min(), nnz_per_row.max(), nnz_per_row.mean(),
        )

        # Check row sums (should be ~1 after normalization)
        row_sums = np.array(P_csr.sum(axis=1)).ravel()
        logger.info(
            "  Row sums: min=%.4f, max=%.4f (should be 1.0)",
            row_sums.min(), row_sums.max(),
        )

    @staticmethod
    def _validate_inputs(A: sp.spmatrix) -> None:
        if not sp.issparse(A):
            raise TypeError(
                f"A must be a scipy sparse matrix, got {type(A).__name__}."
            )
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"A must be square, got shape {A.shape}.")
        if not np.isfinite(A.data).all():
            raise ValueError("A contains NaN or Inf values.")

    def _validate_rhs(self, b: np.ndarray) -> None:
        if not isinstance(b, np.ndarray):
            raise TypeError(
                f"b must be a numpy ndarray, got {type(b).__name__}."
            )
        if b.shape != (self.N,):
            raise ValueError(
                f"b must have shape ({self.N},), got {b.shape}."
            )
        if not np.isfinite(b).all():
            raise ValueError("b contains NaN or Inf values.")