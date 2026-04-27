# -*- coding: utf-8 -*-

"""
ml_solver.py - ML-enhanced AMG solver with learned per-cluster Jacobi weights.

This module provides MLAMGSolver, which extends AMGSolver by replacing the
single global Jacobi damping weight with a per-node weight vector predicted
by a trained MLP (WeightPredictorMLP).

The key difference from the classical AMGSolver:

    Classical:  u += omega * D^-1 * (f - A*u)         [scalar omega]
    ML-AMG:     u += omega_i * D^-1 * (f - A*u)       [vector omega, one per node]

where omega_i is determined by the cluster that node i belongs to,
and each cluster's omega is predicted by the MLP from 7 local features.

Usage
-----
    from amg.ml_solver import MLAMGSolver

    solver = MLAMGSolver(
        A,
        ml_model_path='amg_weight_predictor.pth',
        use_ml_weights=True,
        smoother_weight_fallback=0.7,
    )
    result = solver.solve(b, method='lgmres', tol=1e-10)
"""

import logging
import time
from typing import Callable, Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import gmres, cg, bicgstab, LinearOperator

from .prolongation import build_prolongation
from .smoothers import make_coarse_solver
from .result import SolverResult

logger = logging.getLogger(__name__)

# Optional imports — only required if use_ml_weights=True
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning(
        "PyTorch not found. MLAMGSolver will fall back to scalar smoother_weight."
    )


class MLAMGSolver:
    """
    Two-level AMG solver with ML-predicted per-cluster Jacobi weights.

    Extends the classical AMGSolver by using a trained MLP to predict
    an optimal Jacobi damping weight for each METIS cluster. These
    cluster weights are then broadcast to all fine nodes in the cluster,
    producing a node-wise weight vector used during smoothing.

    If the model cannot be loaded (missing file, PyTorch not installed,
    wrong features), the solver automatically falls back to the scalar
    smoother_weight_fallback value, behaving identically to AMGSolver.

    Parameters
    ----------
    A : sp.spmatrix
        The system matrix (N x N). Must be square and sparse.
    ml_model_path : str, optional
        Path to the trained .pth model file. If None or not found,
        falls back to scalar weights.
    use_ml_weights : bool, optional
        Whether to use ML-predicted weights. Default is True.
    num_clusters : int, optional
        Number of METIS clusters. Default is max(2, N // 1000).
    p_method : str, optional
        Prolongation method: 'binary' or 'smoothed'. Default is 'smoothed'.
    omega : float, optional
        Jacobi damping for prolongation smoothing. Default is 2/3.
    smoothing_passes : int, optional
        Number of Jacobi passes for smoothed P. Default is 1.
    normalize_P : bool, optional
        Normalize rows of P to sum to 1. Default is True.
    clip_P_negatives : bool, optional
        Clip negative P entries to 0. Default is False.
    smoother_steps_pre : int, optional
        Pre-smoothing sweeps per V-cycle. Default is 2.
    smoother_steps_post : int, optional
        Post-smoothing sweeps per V-cycle. Default is 2.
    smoother_weight_fallback : float, optional
        Scalar fallback weight used when ML weights are unavailable.
        Default is 0.7.
    coarse_solver : str, optional
        Coarse-level solver: 'direct' or 'cg'. Default is 'direct'.
    omega_range : tuple of float, optional
        (min, max) range for dynamic scaling of ML predictions.
        Default is (0.7, 0.95).
    """

    def __init__(
        self,
        A: sp.spmatrix,
        ml_model_path: Optional[str] = None,
        use_ml_weights: bool = True,
        num_clusters: Optional[int] = None,
        p_method: str = "smoothed",
        omega: float = 2 / 3,
        smoothing_passes: int = 1,
        normalize_P: bool = True,
        clip_P_negatives: bool = False,
        smoother_steps_pre: int = 2,
        smoother_steps_post: int = 2,
        smoother_weight_fallback: float = 0.7,
        coarse_solver: str = "direct",
        omega_range: tuple = (0.7, 0.95),
    ) -> None:
        self._validate_inputs(A)
        self.A = A.tocsr()
        self.N = A.shape[0]

        self.smoother_steps_pre = smoother_steps_pre
        self.smoother_steps_post = smoother_steps_post
        self.smoother_weight_fallback = smoother_weight_fallback
        self.omega_range = omega_range

        self.n_parts = num_clusters if num_clusters else max(2, self.N // 1000)

        logger.info(
            "Initializing MLAMGSolver: N=%d, n_parts=%d, p_method='%s', "
            "use_ml_weights=%s, coarse_solver='%s'",
            self.N, self.n_parts, p_method, use_ml_weights, coarse_solver,
        )

        # --- Build prolongation (reuses existing build_prolongation) ---
        # NOTE: MLAMGSolver needs access to the METIS membership array
        # to assign per-cluster weights to nodes. We rebuild METIS here
        # to get membership, then pass it to build_prolongation.
        self.membership, self.P = self._build_prolongation_with_membership(
            method=p_method,
            omega=omega,
            smoothing_passes=smoothing_passes,
            normalize_P=normalize_P,
            clip_P_negatives=clip_P_negatives,
        )

        self.A_c = (self.P.T @ self.A @ self.P).tocsr()
        logger.info(
            "Coarse matrix built: shape=%s, nnz=%d", self.A_c.shape, self.A_c.nnz
        )

        self._coarse_solver = make_coarse_solver(self.A_c, method=coarse_solver)

        # --- Load ML weights ---
        self.node_omegas = self._load_ml_weights(
            ml_model_path=ml_model_path,
            use_ml_weights=use_ml_weights,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        b: np.ndarray,
        method: str = "lgmres",
        tol: float = 1e-6,
        max_iter: int = 500,
        callback: Optional[Callable] = None,
    ) -> SolverResult:
        """
        Solve A x = b using the ML-AMG V-cycle as a preconditioner.

        Parameters
        ----------
        b : np.ndarray
            Right-hand side vector of length N.
        method : str, optional
            Outer Krylov method: 'cg', 'gmres', 'bicgstab', 'lgmres'.
            Default is 'lgmres'.
        tol : float, optional
            Convergence tolerance (relative residual). Default is 1e-6.
        max_iter : int, optional
            Maximum number of outer iterations. Default is 500.
        callback : callable, optional
            Optional user callback at each iteration.

        Returns
        -------
        SolverResult
            Dataclass containing solution, convergence info, and timing.
        """
        self._validate_rhs(b)
        if method.lower() not in ("gmres", "cg", "bicgstab", "lgmres"):
            raise ValueError(
                f"Unknown method: '{method}'. "
                "Choose 'gmres', 'cg', 'bicgstab', or 'lgmres'."
            )

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
            if callback is not None:
                callback(
                    iteration=iteration_counter[0], residual=rel_res, x=xk
                )

        def _track_gmres(rk):
            residuals.append(float(rk))
            iteration_counter[0] += 1
            if callback is not None:
                callback(
                    iteration=iteration_counter[0], residual=float(rk), x=None
                )

        logger.info(
            "Starting ML-AMG %s solve (tol=%.1e, max_iter=%d)...",
            method.upper(), tol, max_iter,
        )
        start_time = time.time()

        if method.lower() == "cg":
            x, info = cg(
                self.A, b, M=M_inv, atol=tol,
                callback=_track, maxiter=max_iter,
            )
        elif method.lower() == "bicgstab":
            x, info = bicgstab(
                self.A, b, M=M_inv, atol=tol,
                callback=_track, maxiter=max_iter,
            )
        elif method.lower() == "lgmres":
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
        converged = info == 0

        result = SolverResult(
            x=x,
            converged=converged,
            iterations=len(residuals),
            residual_history=residuals,
            solve_time=solve_time,
            info=info,
            method=f"ml_{method.lower()}",
            tol=tol,
        )

        logger.info(result)
        return result

    def v_cycle(self, f: np.ndarray) -> np.ndarray:
        """
        Perform one ML-AMG V-cycle.

        Uses node-wise Jacobi weights (node_omegas) instead of a
        single scalar weight.

        Parameters
        ----------
        f : np.ndarray
            Fine-level right-hand side vector.

        Returns
        -------
        np.ndarray
            Approximate correction after one V-cycle.
        """
        u = np.zeros_like(f)
        u = self._ml_jacobi(f, u, steps=self.smoother_steps_pre)

        r_c = self.P.T @ (f - self.A @ u)
        c_c = self._coarse_solver.solve(r_c)
        u += self.P @ c_c

        u = self._ml_jacobi(f, u, steps=self.smoother_steps_post)
        return u

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ml_jacobi(
        self, f: np.ndarray, u: np.ndarray, steps: int = 1
    ) -> np.ndarray:
        """
        Node-wise weighted Jacobi smoother.

        u += node_omegas * D^-1 * (f - A*u)

        where node_omegas is a vector of length N, one value per node,
        determined by the cluster the node belongs to.

        Parameters
        ----------
        f : np.ndarray
            Right-hand side vector.
        u : np.ndarray
            Current iterate.
        steps : int
            Number of smoothing sweeps.

        Returns
        -------
        np.ndarray
            Updated iterate.
        """
        D = self.A.diagonal()
        with np.errstate(divide="ignore", invalid="ignore"):
            D_inv = np.where(np.abs(D) > 1e-15, 1.0 / D, 0.0)

        for _ in range(steps):
            residual = f - self.A @ u
            u = u + self.node_omegas * (D_inv * residual)
        return u

    def _load_ml_weights(
        self,
        ml_model_path: Optional[str],
        use_ml_weights: bool,
    ) -> np.ndarray:
        """
        Load ML model and predict per-node omega weights.

        Falls back to scalar smoother_weight_fallback if:
        - use_ml_weights is False
        - ml_model_path is None
        - PyTorch is not installed
        - Model file not found
        - Any other exception during loading/inference

        Returns
        -------
        np.ndarray
            Array of shape (N,) with per-node omega values.
        """
        fallback = np.full(self.N, self.smoother_weight_fallback)

        if not use_ml_weights or ml_model_path is None:
            logger.info(
                "ML weights disabled. Using scalar fallback omega=%.2f.",
                self.smoother_weight_fallback,
            )
            return fallback

        if not _TORCH_AVAILABLE:
            logger.warning(
                "PyTorch not available. Falling back to scalar omega=%.2f.",
                self.smoother_weight_fallback,
            )
            return fallback

        try:
            from .ml_weight_predictor import load_model
            from .ml_features import extract_cluster_features

            model = load_model(ml_model_path)

            feats = extract_cluster_features(
                self.A, self.membership, self.n_parts
            )

            with torch.no_grad():
                raw_preds = (
                    model(
                        torch.tensor(feats, dtype=torch.float32)
                    )
                    .numpy()
                    .flatten()
                )

            # Dynamic scaling: map raw output in (0,1) to omega_range
            p_min, p_max = raw_preds.min(), raw_preds.max()
            omega_lo, omega_hi = self.omega_range
            if p_max > p_min:
                normalized = (raw_preds - p_min) / (p_max - p_min)
                cluster_omegas = omega_lo + normalized * (omega_hi - omega_lo)
            else:
                cluster_omegas = np.full(self.n_parts, (omega_lo + omega_hi) / 2)

            node_omegas = cluster_omegas[self.membership]

            logger.info(
                "ML weights loaded. Node omega range: [%.3f, %.3f].",
                node_omegas.min(), node_omegas.max(),
            )
            return node_omegas

        except FileNotFoundError:
            logger.warning(
                "Model file '%s' not found. Falling back to scalar omega=%.2f.",
                ml_model_path, self.smoother_weight_fallback,
            )
            return fallback
        except Exception as e:
            logger.warning(
                "ML weight loading failed (%s). Falling back to scalar omega=%.2f.",
                str(e), self.smoother_weight_fallback,
            )
            return fallback

    def _build_prolongation_with_membership(
        self,
        method: str,
        omega: float,
        smoothing_passes: int,
        normalize_P: bool,
        clip_P_negatives: bool,
    ) -> tuple:
        """
        Build METIS clusters and prolongation matrix.

        Returns both the membership array (needed for weight assignment)
        and the prolongation matrix P.

        Returns
        -------
        membership : np.ndarray
            Integer array of shape (N,) mapping nodes to cluster IDs.
        P : sp.csr_matrix
            Prolongation matrix of shape (N, n_parts).
        """
        import pymetis

        is_symmetric = (self.A != self.A.T).nnz == 0
        adj_mat = self.A if is_symmetric else (self.A + self.A.T)

        logger.info(
            "Partitioning graph into %d clusters (METIS)...", self.n_parts
        )
        adjacency = pymetis.CSRAdjacency(adj_mat.indptr, adj_mat.indices)
        _, membership_list = pymetis.part_graph(
            self.n_parts, adjacency=adjacency
        )
        membership = np.array(membership_list)

        # Build tentative P from membership
        row_indices = np.arange(self.N)
        P_tentative = sp.csr_matrix(
            (np.ones(self.N), (row_indices, membership)),
            shape=(self.N, self.n_parts),
        )

        # Apply smoothing if requested
        if method == "binary":
            P = P_tentative
        elif method == "smoothed":
            P = self._smooth_prolongation(
                P_tentative,
                omega=omega,
                smoothing_passes=smoothing_passes,
                normalize_P=normalize_P,
                clip_P_negatives=clip_P_negatives,
            )
        else:
            raise ValueError(
                f"Unknown p_method: '{method}'. Choose 'binary' or 'smoothed'."
            )

        logger.info(
            "Prolongation built: shape=%s, nnz=%d", P.shape, P.nnz
        )
        return membership, P

    def _smooth_prolongation(
        self,
        P_tent: sp.csr_matrix,
        omega: float,
        smoothing_passes: int,
        normalize_P: bool,
        clip_P_negatives: bool,
    ) -> sp.csr_matrix:
        """Apply Jacobi smoothing to tentative prolongation."""
        D = self.A.diagonal()
        with np.errstate(divide="ignore", invalid="ignore"):
            D_safe = np.where(np.abs(D) < 1e-15, 1.0, D)
        D_inv = sp.diags(1.0 / D_safe)

        I = sp.eye(self.N)
        Smoother = I - omega * D_inv @ self.A

        P = P_tent.copy().tocsr()
        for _ in range(max(1, smoothing_passes)):
            P = (Smoother @ P).tocsr()

        if clip_P_negatives:
            P.data = np.maximum(P.data, 0.0)
            P.eliminate_zeros()

        if normalize_P or clip_P_negatives:
            row_sums = np.array(P.sum(axis=1)).flatten()
            zero_rows = np.abs(row_sums) < 1e-15
            row_sums[zero_rows] = 1.0
            P = sp.diags(1.0 / row_sums) @ P
            P = P.tocsr()

        return P

    @staticmethod
    def _validate_inputs(A: sp.spmatrix) -> None:
        if not sp.issparse(A):
            raise TypeError(
                f"A must be a scipy sparse matrix, got {type(A).__name__}."
            )
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(
                f"A must be square, got shape {A.shape}."
            )
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