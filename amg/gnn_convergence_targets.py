"""
gnn_solver.py - AMG solver with GNN-predicted prolongation operator (GNN-P).

This is the most sophisticated ML-AMG approach in the library.
Instead of an MLP operating on scalar node features, a Graph Neural
Network (GNN) sees the actual sparsity graph of A, performing message
passing over the local neighbourhood of each node before predicting
per-node interpolation strengths.

Key advantage over MLProlongationSolver:
    The GNN captures graph topology — it knows not just what a node's
    local matrix entries look like, but how its neighbours relate to
    each other and to the cluster boundaries. This is exactly the
    information that determines good AMG interpolation.

Two GNN architectures are supported (set via gnn_architecture):
    'sage'      - EdgeWeightedSAGEConv (simpler, faster)
    'attention' - AttentionSAGEConv with multi-head attention (recommended)

Two prolongation methods are supported (set via prolongation_method):
    'smoothed'  - Jacobi-style: P_ij = -alpha_i * D_i^-1 * A_ij
    'direct'    - Threshold-based: includes off-cluster j if score > threshold

P construction is fully vectorized — no Python loops over nodes.

Usage
-----
    from amg.gnn_solver import GNNProlongationSolver

    solver = GNNProlongationSolver(
        A,
        gnn_model_path='gnn_model_attn.pth',
        prolongation_method='smoothed',
    )
    result = solver.solve(b, method='lgmres', tol=1e-10)
"""

import logging
import time
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, lgmres, cg, bicgstab, gmres

from .smoothers import make_coarse_solver
from .result import SolverResult

logger = logging.getLogger(__name__)

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch not found. GNNProlongationSolver will not work.")


class GNNProlongationSolver:
    """
    Two-level AMG solver with GNN-predicted prolongation matrix.

    The GNN performs message passing over the sparsity graph of A,
    capturing the local topology around each node before predicting
    per-node interpolation strengths s_i in (0,1). These strengths
    control how aggressively each fine node interpolates from
    neighbouring clusters when building P.

    Parameters
    ----------
    A : sp.spmatrix
        System matrix (N x N), square and sparse.
    gnn_model_path : str
        Path to trained .pth checkpoint (saved by training/train_gnn.py).
    num_clusters : int, optional
        Number of METIS clusters. Default: max(2, N // 1000).
    prolongation_method : str, optional
        'smoothed' or 'direct'. Default: 'smoothed'.
    smoother_steps_pre : int, optional
        Pre-smoothing Jacobi sweeps per V-cycle. Default: 2.
    smoother_steps_post : int, optional
        Post-smoothing Jacobi sweeps per V-cycle. Default: 2.
    smoother_weight : float, optional
        Jacobi damping weight. Default: 0.7.
    coarse_solver : str, optional
        'direct' (LU factorization) or 'cg'. Default: 'direct'.
    alpha_scale : float, optional
        Scaling for off-cluster weights in smoothed P. Default: 0.67.
    direct_threshold_base : float, optional
        Base threshold for direct P method. Default: 0.15.
    """

    def __init__(
        self,
        A: sp.spmatrix,
        gnn_model_path: str,
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
        self.A  = A.tocsr()
        self.N  = A.shape[0]
        self.smoother_steps_pre   = smoother_steps_pre
        self.smoother_steps_post  = smoother_steps_post
        self.smoother_weight      = smoother_weight
        self.alpha_scale          = alpha_scale
        self.direct_threshold_base = direct_threshold_base

        self.n_parts = num_clusters if num_clusters else max(2, self.N // 1000)

        logger.info(
            "Initializing GNNProlongationSolver: N=%d, n_parts=%d, "
            "method='%s', coarse_solver='%s'",
            self.N, self.n_parts, prolongation_method, coarse_solver,
        )

        # 1. METIS partitioning
        self.membership = self._run_metis()

        # 2. Build graph tensors for GNN inference
        x, edge_index, edge_attr = self._build_graph_tensors()

        # 3. GNN inference → node strengths
        node_strengths = self._predict_node_strengths(
            gnn_model_path, x, edge_index, edge_attr
        )

        # 4. Build P (vectorized, no Python loops)
        t0 = time.time()
        if prolongation_method == "smoothed":
            self.P = self._build_P_smoothed(node_strengths)
        elif prolongation_method == "direct":
            self.P = self._build_P_direct(node_strengths)
        else:
            raise ValueError(
                f"Unknown prolongation_method '{prolongation_method}'. "
                "Use 'smoothed' or 'direct'."
            )
        logger.info(
            "P built in %.1fs: shape=%s, nnz=%d",
            time.time() - t0, self.P.shape, self.P.nnz,
        )

        # 5. Diagnostics
        self._log_P_diagnostics()

        # 6. Coarse matrix
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
        Solve A x = b using GNN-P V-cycle as preconditioner.

        Parameters
        ----------
        b : np.ndarray
            RHS vector of length N.
        method : str
            Krylov method: 'cg', 'gmres', 'bicgstab', 'lgmres'.
        tol : float
            Convergence tolerance.
        max_iter : int
            Maximum iterations.

        Returns
        -------
        SolverResult
        """
        self._validate_rhs(b)

        M = LinearOperator(
            shape=(self.N, self.N),
            matvec=self.v_cycle,
            dtype=self.A.dtype,
        )

        residuals = []

        def _track(xk):
            residuals.append(
                np.linalg.norm(b - self.A @ xk) / (np.linalg.norm(b) + 1e-300)
            )

        def _track_gmres(rk):
            residuals.append(float(rk))

        logger.info(
            "Starting GNN-P %s solve (tol=%.1e, max_iter=%d)...",
            method.upper(), tol, max_iter,
        )
        t0 = time.time()

        m = method.lower()
        if m == "cg":
            x, info = cg(self.A, b, M=M, atol=tol,
                         callback=_track, maxiter=max_iter)
        elif m == "bicgstab":
            x, info = bicgstab(self.A, b, M=M, atol=tol,
                               callback=_track, maxiter=max_iter)
        elif m == "lgmres":
            x, info = lgmres(self.A, b, M=M, atol=tol,
                             callback=_track, maxiter=max_iter,
                             inner_m=30, outer_k=3)
        else:
            x, info = gmres(self.A, b, M=M, atol=tol,
                            callback=_track_gmres,
                            callback_type="legacy",
                            maxiter=max_iter)

        result = SolverResult(
            x=x,
            converged=(info == 0),
            iterations=len(residuals),
            residual_history=residuals,
            solve_time=time.time() - t0,
            info=info,
            method=f"gnnp_{m}",
            tol=tol,
        )
        logger.info(result)
        return result

    def v_cycle(self, f: np.ndarray) -> np.ndarray:
        """One GNN-P V-cycle (used as preconditioner action)."""
        u = np.zeros_like(f)
        u = self._jacobi(f, u, self.smoother_steps_pre)
        r_c = self.P.T @ (f - self.A @ u)
        u  += self.P @ self._coarse_solver.solve(r_c)
        u   = self._jacobi(f, u, self.smoother_steps_post)
        return u

    # ------------------------------------------------------------------
    # Graph construction for GNN inference
    # ------------------------------------------------------------------

    def _build_graph_tensors(self) -> Tuple:
        """
        Build node feature matrix and edge tensors from A and membership.

        Node features (10 per node):
            0  diagonal A_ii
            1  row sum |A_i*|
            2  degree
            3  max |A_ij|
            4  mean |A_ij|
            5  cluster size
            6  same-cluster neighbor count
            7  different-cluster neighbor count
            8  mean diagonal of neighbors
            9  mean row sum of neighbors

        Edge attributes (3 per edge):
            0  |a_ij|
            1  |a_ij| / |a_ii|   normalized coupling
            2  same_cluster flag  (1.0 if membership[i] == membership[j])
        """
        if not _TORCH_AVAILABLE:
            return None, None, None

        logger.info("Building graph tensors for GNN inference (N=%d)...", self.N)
        t0 = time.time()

        A   = self.A
        N   = self.N
        mem = self.membership

        diag     = A.diagonal()
        abs_A    = abs(A)
        row_sum  = np.array(abs_A.sum(axis=1)).flatten()
        degree   = np.diff(A.indptr)

        # max/mean per row (vectorized via reduceat)
        abs_data = np.abs(A.data)
        indptr   = A.indptr
        nonempty = np.where(indptr[1:] > indptr[:-1])[0]
        max_abs  = np.zeros(N)
        mean_abs = np.zeros(N)
        for i in nonempty:
            s, e = indptr[i], indptr[i+1]
            v = abs_data[s:e]
            max_abs[i]  = v.max()
            mean_abs[i] = v.mean()

        # cluster sizes
        cluster_size = np.zeros(N)
        for c in range(self.n_parts):
            idx = np.where(mem == c)[0]
            cluster_size[idx] = len(idx)

        # neighbor counts and mean neighbor stats (vectorized)
        A_coo   = A.tocoo()
        row_idx = A_coo.row
        col_idx = A_coo.col

        same_mask = mem[row_idx] == mem[col_idx]
        neigh_same  = np.zeros(N)
        neigh_other = np.zeros(N)
        np.add.at(neigh_same,  row_idx[same_mask],  1)
        np.add.at(neigh_other, row_idx[~same_mask], 1)

        mean_neigh_diag   = np.zeros(N)
        mean_neigh_rowsum = np.zeros(N)
        np.add.at(mean_neigh_diag,   row_idx, diag[col_idx])
        np.add.at(mean_neigh_rowsum, row_idx, row_sum[col_idx])
        safe_deg = np.maximum(degree, 1)
        mean_neigh_diag   /= safe_deg
        mean_neigh_rowsum /= safe_deg
        isolated = degree == 0
        mean_neigh_diag[isolated]   = diag[isolated]
        mean_neigh_rowsum[isolated] = row_sum[isolated]

        node_features = np.column_stack([
            diag, row_sum, degree, max_abs, mean_abs,
            cluster_size, neigh_same, neigh_other,
            mean_neigh_diag, mean_neigh_rowsum,
        ]).astype(np.float32)

        # Edge index and attributes (exclude diagonal)
        off_mask   = row_idx != col_idx
        src        = row_idx[off_mask]
        dst        = col_idx[off_mask]
        abs_vals   = np.abs(A_coo.data[off_mask])
        norm_vals  = abs_vals / (np.abs(diag[src]) + 1e-12)
        same_flags = (mem[src] == mem[dst]).astype(np.float32)

        edge_index = np.array([src, dst], dtype=np.int64)
        edge_attr  = np.column_stack(
            [abs_vals, norm_vals, same_flags]
        ).astype(np.float32)

        logger.info(
            "Graph tensors built in %.1fs: nodes=%d, edges=%d",
            time.time() - t0, N, len(src),
        )

        x_t  = torch.tensor(node_features, dtype=torch.float32)
        ei_t = torch.tensor(edge_index,    dtype=torch.long)
        ea_t = torch.tensor(edge_attr,     dtype=torch.float32)
        return x_t, ei_t, ea_t

    # ------------------------------------------------------------------
    # GNN inference
    # ------------------------------------------------------------------

    def _predict_node_strengths(
        self,
        gnn_model_path: str,
        x: "torch.Tensor",
        edge_index: "torch.Tensor",
        edge_attr: "torch.Tensor",
    ) -> np.ndarray:
        """
        Run GNN forward pass to get per-node interpolation strengths.
        Falls back to uniform 0.5 if model cannot be loaded.
        """
        fallback = np.full(self.N, 0.5)

        if not _TORCH_AVAILABLE:
            logger.warning("PyTorch unavailable. Using uniform strength=0.5.")
            return fallback

        try:
            from .gnn_model import load_gnn_model

            logger.info("Loading GNN model from %s...", gnn_model_path)
            model, norm = load_gnn_model(gnn_model_path)

            # Normalize features
            X_mean = torch.tensor(norm["X_mean"], dtype=torch.float32)
            X_std  = torch.tensor(norm["X_std"],  dtype=torch.float32)
            e_mean = torch.tensor(norm["edge_mean"], dtype=torch.float32)
            e_std  = torch.tensor(norm["edge_std"],  dtype=torch.float32)

            x_norm  = (x  - X_mean) / (X_std  + 1e-8)
            ea_norm = (edge_attr - e_mean) / (e_std + 1e-8)

            with torch.no_grad():
                raw = model(x_norm, edge_index, ea_norm).numpy().flatten()

            node_strengths = 1.0 / (1.0 + np.exp(-raw))   # sigmoid

            logger.info(
                "GNN-P strengths: min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
                node_strengths.min(), node_strengths.max(),
                node_strengths.mean(), node_strengths.std(),
            )
            return node_strengths

        except FileNotFoundError:
            logger.warning(
                "GNN model file '%s' not found. Using uniform strength=0.5.",
                gnn_model_path,
            )
            return fallback
        except Exception as e:
            logger.warning(
                "GNN model loading failed (%s). Using uniform strength=0.5.", e
            )
            return fallback

    # ------------------------------------------------------------------
    # P construction (vectorized)
    # ------------------------------------------------------------------

    def _build_P_smoothed(self, node_strengths: np.ndarray) -> sp.csr_matrix:
        """
        Smoothed prolongation — same formula as MLProlongationSolver
        but now with GNN-predicted strengths.

        P_i,own = 1 - sum_{j: nc != own}(-alpha_i * D_i^-1 * A_ij)
        P_i,nc  = -alpha_i * D_i^-1 * A_ij  for j in cluster nc != own
        """
        A   = self.A
        mem = self.membership
        D_inv = 1.0 / (A.diagonal() + 1e-12)
        alpha = self.alpha_scale * node_strengths

        A_coo       = A.tocoo()
        row_idx     = A_coo.row
        col_idx     = A_coo.col
        vals        = A_coo.data
        col_cluster = mem[col_idx]
        row_cluster = mem[row_idx]

        off_mask    = col_cluster != row_cluster
        off_rows    = row_idx[off_mask]
        off_cols    = col_cluster[off_mask]
        off_weights = -alpha[off_rows] * D_inv[off_rows] * vals[off_mask]

        own_adjustment = np.zeros(self.N)
        np.add.at(own_adjustment, off_rows, off_weights)

        all_rows = np.concatenate([off_rows,    np.arange(self.N)])
        all_cols = np.concatenate([off_cols,    mem])
        all_data = np.concatenate([off_weights, 1.0 - own_adjustment])

        P_raw = sp.csr_matrix(
            (all_data, (all_rows, all_cols)),
            shape=(self.N, self.n_parts),
        )
        row_sums = np.array(P_raw.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        return (sp.diags(1.0 / row_sums) @ P_raw).tocsr()

    def _build_P_direct(self, node_strengths: np.ndarray) -> sp.csr_matrix:
        """
        Threshold-based prolongation — includes off-cluster neighbours
        whose normalized coupling exceeds an adaptive threshold.

        score_ij   = s_i * |A_ij| / |A_ii|
        threshold_i = base * (1 - s_i * 0.5)
        """
        A   = self.A
        mem = self.membership
        abs_diag = np.abs(A.diagonal()) + 1e-12

        A_coo    = A.tocoo()
        row_idx  = A_coo.row
        col_idx  = A_coo.col
        abs_vals = np.abs(A_coo.data)

        col_cluster = mem[col_idx]
        row_cluster = mem[row_idx]
        off_mask    = (col_cluster != row_cluster) & (row_idx != col_idx)

        off_rows   = row_idx[off_mask]
        off_cols   = col_cluster[off_mask]
        off_avals  = abs_vals[off_mask]

        scores     = node_strengths[off_rows] * off_avals / abs_diag[off_rows]
        thresholds = self.direct_threshold_base * (
            1.0 - node_strengths[off_rows] * 0.5
        )
        keep = scores > thresholds

        all_rows = np.concatenate([off_rows[keep], np.arange(self.N)])
        all_cols = np.concatenate([off_cols[keep], mem])
        all_data = np.concatenate([scores[keep],   np.ones(self.N)])

        P_raw = sp.csr_matrix(
            (all_data, (all_rows, all_cols)),
            shape=(self.N, self.n_parts),
        )
        row_sums = np.array(P_raw.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        return (sp.diags(1.0 / row_sums) @ P_raw).tocsr()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_metis(self) -> np.ndarray:
        import pymetis
        is_sym = (self.A != self.A.T).nnz == 0
        adj    = self.A if is_sym else (self.A + self.A.T)
        logger.info("Partitioning into %d clusters (METIS)...", self.n_parts)
        _, mem = pymetis.part_graph(
            self.n_parts,
            adjacency=pymetis.CSRAdjacency(adj.indptr, adj.indices),
        )
        return np.array(mem)

    def _jacobi(self, f, u, steps):
        D = self.A.diagonal()
        D_inv = np.where(np.abs(D) > 1e-15, 1.0 / D, 0.0)
        for _ in range(steps):
            u = u + self.smoother_weight * D_inv * (f - self.A @ u)
        return u

    def _log_P_diagnostics(self) -> None:
        nnz_per_row = np.diff(self.P.indptr)
        n_inj = (nnz_per_row == 1).sum()
        n_int = (nnz_per_row >= 2).sum()
        logger.info("P diagnostics:")
        logger.info(
            "  Pure injection (1 nnz): %d / %d (%.1f%%)",
            n_inj, self.N, 100 * n_inj / self.N,
        )
        logger.info(
            "  Interpolation (2+ nnz): %d / %d (%.1f%%)",
            n_int, self.N, 100 * n_int / self.N,
        )
        logger.info(
            "  Row nnz: min=%d, max=%d, mean=%.2f",
            nnz_per_row.min(), nnz_per_row.max(), nnz_per_row.mean(),
        )
        row_sums = np.array(self.P.sum(axis=1)).ravel()
        logger.info(
            "  Row sums: min=%.4f, max=%.4f (should be ~1.0)",
            row_sums.min(), row_sums.max(),
        )

    @staticmethod
    def _validate_inputs(A):
        if not sp.issparse(A):
            raise TypeError(f"A must be sparse, got {type(A).__name__}.")
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"A must be square, got shape {A.shape}.")
        if not np.isfinite(A.data).all():
            raise ValueError("A contains NaN or Inf.")

    def _validate_rhs(self, b):
        if not isinstance(b, np.ndarray):
            raise TypeError(f"b must be ndarray, got {type(b).__name__}.")
        if b.shape != (self.N,):
            raise ValueError(f"b must have shape ({self.N},), got {b.shape}.")
        if not np.isfinite(b).all():
            raise ValueError("b contains NaN or Inf.")