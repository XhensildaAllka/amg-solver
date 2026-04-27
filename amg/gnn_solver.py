"""
gnn_solver.py - AMG solver with GNN-predicted prolongation operator (GNN-P).

Memory optimisations for large matrices (1M+ nodes):
  1. GNN model is loaded FIRST before building any graph tensors.
  2. GNN inference runs in mini-batches (default 50k nodes per batch).
     Each batch builds only the local subgraph, runs forward, then frees tensors.
     Peak RAM is O(batch_size) not O(N).
  3. Tensors are explicitly deleted + gc.collect() after inference.

Two architectures: 'sage' or 'attention' (auto-detected from checkpoint).
Two prolongation methods: 'smoothed' or 'direct'.

Usage
-----
    from amg.gnn_solver import GNNProlongationSolver

    solver = GNNProlongationSolver(
        A,
        gnn_model_path='training/models/gnn_model_attn_proxy.pth',
        prolongation_method='smoothed',
    )
    result = solver.solve(b, method='lgmres', tol=1e-10)
"""

import gc
import logging
import time
from typing import Optional

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
    logger.warning("PyTorch not found. GNNProlongationSolver will use uniform strengths.")


class GNNProlongationSolver:
    """
    Two-level AMG solver with GNN-predicted prolongation matrix.

    Parameters
    ----------
    A : sp.spmatrix
        System matrix (N x N), square and sparse.
    gnn_model_path : str
        Path to trained .pth checkpoint (from training/train_gnn.py).
    num_clusters : int, optional
        Number of METIS clusters. Default: max(2, N // 1000).
    prolongation_method : str, optional
        'smoothed' or 'direct'. Default: 'smoothed'.
    smoother_steps_pre : int, optional
        Pre-smoothing Jacobi sweeps. Default: 2.
    smoother_steps_post : int, optional
        Post-smoothing Jacobi sweeps. Default: 2.
    smoother_weight : float, optional
        Jacobi damping weight. Default: 0.7.
    coarse_solver : str, optional
        'direct' or 'cg'. Default: 'direct'.
    alpha_scale : float, optional
        Scaling for smoothed P off-cluster weights. Default: 0.67.
    direct_threshold_base : float, optional
        Base threshold for direct P. Default: 0.15.
    inference_batch_size : int, optional
        Nodes per mini-batch during GNN inference. Default: 50000.
        Reduce if you run out of RAM, increase for speed.
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
        inference_batch_size: int = 50_000,
    ) -> None:
        self._validate_inputs(A)
        self.A                     = A.tocsr()
        self.N                     = A.shape[0]
        self.smoother_steps_pre    = smoother_steps_pre
        self.smoother_steps_post   = smoother_steps_post
        self.smoother_weight       = smoother_weight
        self.alpha_scale           = alpha_scale
        self.direct_threshold_base = direct_threshold_base
        self.inference_batch_size  = inference_batch_size
        self.n_parts               = num_clusters if num_clusters else max(2, self.N // 1000)

        logger.info(
            "Initializing GNNProlongationSolver: N=%d, n_parts=%d, "
            "method='%s', coarse_solver='%s', batch_size=%d",
            self.N, self.n_parts, prolongation_method,
            coarse_solver, inference_batch_size,
        )

        # Step 1 — Load model FIRST while RAM is still free
        model, norm = self._load_model(gnn_model_path)

        # Step 2 — METIS partitioning
        self.membership = self._run_metis()

        # Step 3 — Mini-batch GNN inference (O(batch_size) RAM peak)
        node_strengths = self._predict_batched(model, norm)

        # Free model from memory before building P
        del model, norm
        gc.collect()

        # Step 4 — Build P (fully vectorized)
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
        logger.info("P built in %.1fs: shape=%s, nnz=%d",
                    time.time() - t0, self.P.shape, self.P.nnz)

        # Step 5 — Diagnostics
        self._log_P_diagnostics()

        # Step 6 — Coarse matrix and solver
        self.A_c = (self.P.T @ self.A @ self.P).tocsr()
        logger.info("Coarse matrix: shape=%s, nnz=%d", self.A_c.shape, self.A_c.nnz)
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
        """Solve A x = b using GNN-P V-cycle as preconditioner."""
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

        logger.info("Starting GNN-P %s solve (tol=%.1e, max_iter=%d)...",
                    method.upper(), tol, max_iter)
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
        """One GNN-P V-cycle (preconditioner action)."""
        u  = np.zeros_like(f)
        u  = self._jacobi(f, u, self.smoother_steps_pre)
        r_c = self.P.T @ (f - self.A @ u)
        u  += self.P @ self._coarse_solver.solve(r_c)
        u   = self._jacobi(f, u, self.smoother_steps_post)
        return u

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, gnn_model_path: str):
        """Load model FIRST before any graph tensors are built."""
        if not _TORCH_AVAILABLE:
            logger.warning("PyTorch unavailable. Will use uniform strength=0.5.")
            return None, None
        try:
            from .gnn_model import load_gnn_model
            logger.info("Loading GNN model from %s...", gnn_model_path)
            model, norm = load_gnn_model(gnn_model_path)
            logger.info("GNN model loaded.")
            return model, norm
        except FileNotFoundError:
            logger.warning("GNN model '%s' not found. Using uniform strength=0.5.",
                           gnn_model_path)
            return None, None
        except Exception as e:
            logger.warning("GNN model loading failed (%s). Using uniform strength=0.5.", e)
            return None, None

    # ------------------------------------------------------------------
    # Mini-batch GNN inference
    # ------------------------------------------------------------------

    def _predict_batched(self, model, norm) -> np.ndarray:
        """
        Run GNN inference in mini-batches to limit peak RAM.

        For each batch of `inference_batch_size` nodes:
          - Build local subgraph (edges within the batch only)
          - Normalize features
          - Run GNN forward pass
          - Store strengths, delete tensors, gc.collect()
        """
        fallback = np.full(self.N, 0.5, dtype=np.float32)

        if model is None or not _TORCH_AVAILABLE:
            logger.info("Using uniform node strength=0.5.")
            return fallback

        n_batches = int(np.ceil(self.N / self.inference_batch_size))
        logger.info(
            "Mini-batch GNN inference: N=%d, batch_size=%d, n_batches=%d",
            self.N, self.inference_batch_size, n_batches,
        )

        # Precompute node features — O(N * 10), small enough to keep in RAM
        node_features = self._compute_node_features()

        node_strengths = np.zeros(self.N, dtype=np.float32)
        diag           = np.abs(self.A.diagonal())
        mem            = self.membership
        t0             = time.time()

        for batch_idx in range(n_batches):
            bs = batch_idx * self.inference_batch_size
            be = min(bs + self.inference_batch_size, self.N)

            # Build edges entirely within this batch
            rows_list = []
            cols_list = []
            attr_list = []

            for i in range(bs, be):
                s, e = self.A.indptr[i], self.A.indptr[i + 1]
                for k in range(s, e):
                    j = self.A.indices[k]
                    if i == j or j < bs or j >= be:
                        continue
                    li      = i - bs
                    lj      = j - bs
                    aij     = abs(self.A.data[k])
                    norm_ij = aij / (diag[i] + 1e-12)
                    same    = float(mem[i] == mem[j])
                    rows_list.append(li)
                    cols_list.append(lj)
                    attr_list.append([aij, norm_ij, same])

            if len(rows_list) == 0:
                node_strengths[bs:be] = 0.5
                continue

            x_batch = node_features[bs:be]
            x_norm  = (x_batch - norm["X_mean"]) / (norm["X_std"] + 1e-8)
            ea      = np.array(attr_list, dtype=np.float32)
            ea_norm = (ea - norm["edge_mean"]) / (norm["edge_std"] + 1e-8)
            ei      = np.array([rows_list, cols_list], dtype=np.int64)

            x_t  = torch.tensor(x_norm,   dtype=torch.float32)
            ei_t = torch.tensor(ei,        dtype=torch.long)
            ea_t = torch.tensor(ea_norm,   dtype=torch.float32)

            with torch.no_grad():
                raw = model(x_t, ei_t, ea_t).numpy().flatten()

            node_strengths[bs:be] = 1.0 / (1.0 + np.exp(-raw))

            # Free tensors immediately
            del x_t, ei_t, ea_t, x_norm, ea_norm, ea, ei, raw
            gc.collect()

            if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
                pct = 100 * (batch_idx + 1) / n_batches
                logger.info("  Batch %d/%d (%.0f%%)  elapsed=%.1fs",
                            batch_idx + 1, n_batches, pct, time.time() - t0)

        logger.info(
            "GNN inference done in %.1fs — "
            "strengths: min=%.4f max=%.4f mean=%.4f",
            time.time() - t0,
            node_strengths.min(), node_strengths.max(), node_strengths.mean(),
        )
        return node_strengths

    # ------------------------------------------------------------------
    # Node feature computation
    # ------------------------------------------------------------------

    def _compute_node_features(self) -> np.ndarray:
        """10 AMG node features, same as mlp_features.py."""
        logger.info("Computing node features (N=%d)...", self.N)
        A   = self.A
        N   = self.N
        mem = self.membership

        diag    = A.diagonal()
        abs_A   = abs(A)
        row_sum = np.array(abs_A.sum(axis=1)).flatten()
        degree  = np.diff(A.indptr)

        max_abs  = np.zeros(N)
        mean_abs = np.zeros(N)
        for i in np.where(degree > 0)[0]:
            s, e = A.indptr[i], A.indptr[i + 1]
            v = np.abs(A.data[s:e])
            max_abs[i]  = v.max()
            mean_abs[i] = v.mean()

        cluster_size = np.zeros(N)
        for c in range(self.n_parts):
            idx = np.where(mem == c)[0]
            cluster_size[idx] = len(idx)

        A_coo   = A.tocoo()
        row_idx = A_coo.row
        col_idx = A_coo.col

        same_mask   = mem[row_idx] == mem[col_idx]
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

        return np.column_stack([
            diag, row_sum, degree, max_abs, mean_abs,
            cluster_size, neigh_same, neigh_other,
            mean_neigh_diag, mean_neigh_rowsum,
        ]).astype(np.float32)

    # ------------------------------------------------------------------
    # P construction (vectorized)
    # ------------------------------------------------------------------

    def _build_P_smoothed(self, node_strengths: np.ndarray) -> sp.csr_matrix:
        A   = self.A
        mem = self.membership
        D_inv = 1.0 / (A.diagonal() + 1e-12)
        alpha = self.alpha_scale * node_strengths

        A_coo       = A.tocoo()
        row_idx     = A_coo.row
        col_idx     = A_coo.col
        vals        = A_coo.data
        off_mask    = mem[col_idx] != mem[row_idx]
        off_rows    = row_idx[off_mask]
        off_cols    = mem[col_idx[off_mask]]
        off_weights = -alpha[off_rows] * D_inv[off_rows] * vals[off_mask]

        own_adj = np.zeros(self.N)
        np.add.at(own_adj, off_rows, off_weights)

        all_rows = np.concatenate([off_rows,    np.arange(self.N)])
        all_cols = np.concatenate([off_cols,    mem])
        all_data = np.concatenate([off_weights, 1.0 - own_adj])

        P_raw    = sp.csr_matrix((all_data, (all_rows, all_cols)),
                                 shape=(self.N, self.n_parts))
        row_sums = np.array(P_raw.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        return (sp.diags(1.0 / row_sums) @ P_raw).tocsr()

    def _build_P_direct(self, node_strengths: np.ndarray) -> sp.csr_matrix:
        A   = self.A
        mem = self.membership
        abs_diag = np.abs(A.diagonal()) + 1e-12

        A_coo    = A.tocoo()
        row_idx  = A_coo.row
        col_idx  = A_coo.col
        abs_vals = np.abs(A_coo.data)

        off_mask   = (mem[col_idx] != mem[row_idx]) & (row_idx != col_idx)
        off_rows   = row_idx[off_mask]
        off_cols   = mem[col_idx[off_mask]]
        off_avals  = abs_vals[off_mask]

        scores     = node_strengths[off_rows] * off_avals / abs_diag[off_rows]
        thresholds = self.direct_threshold_base * (1.0 - node_strengths[off_rows] * 0.5)
        keep       = scores > thresholds

        all_rows = np.concatenate([off_rows[keep], np.arange(self.N)])
        all_cols = np.concatenate([off_cols[keep], mem])
        all_data = np.concatenate([scores[keep],   np.ones(self.N)])

        P_raw    = sp.csr_matrix((all_data, (all_rows, all_cols)),
                                 shape=(self.N, self.n_parts))
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
        D     = self.A.diagonal()
        D_inv = np.where(np.abs(D) > 1e-15, 1.0 / D, 0.0)
        for _ in range(steps):
            u = u + self.smoother_weight * D_inv * (f - self.A @ u)
        return u

    def _log_P_diagnostics(self) -> None:
        nnz_per_row = np.diff(self.P.indptr)
        n_inj = (nnz_per_row == 1).sum()
        n_int = (nnz_per_row >= 2).sum()
        logger.info("P diagnostics:")
        logger.info("  Pure injection (1 nnz): %d / %d (%.1f%%)",
                    n_inj, self.N, 100 * n_inj / self.N)
        logger.info("  Interpolation (2+ nnz): %d / %d (%.1f%%)",
                    n_int, self.N, 100 * n_int / self.N)
        logger.info("  Row nnz: min=%d, max=%d, mean=%.2f",
                    nnz_per_row.min(), nnz_per_row.max(), nnz_per_row.mean())
        row_sums = np.array(self.P.sum(axis=1)).ravel()
        logger.info("  Row sums: min=%.4f, max=%.4f (should be ~1.0)",
                    row_sums.min(), row_sums.max())

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