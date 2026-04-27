"""
solver.py - Main AMGSolver class.

Implements a two-level Algebraic Multigrid (AMG) solver using:
  - METIS graph partitioning for aggregation
  - Optional Jacobi-smoothed prolongation
  - Weighted Jacobi pre/post smoothing
  - GMRES or CG as the outer Krylov accelerator
"""

import logging
import time
from typing import Callable, Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import gmres, cg, bicgstab, lgmres, LinearOperator
from .prolongation import build_prolongation
from .smoothers import weighted_jacobi, make_coarse_solver
from .result import SolverResult

logger = logging.getLogger(__name__)


class AMGSolver:
    """
    Two-level Algebraic Multigrid (AMG) solver with METIS aggregation.

    The solver builds a two-grid hierarchy:
      - **Fine level**: the original N x N system A.
      - **Coarse level**: A_c = P^T A P, where P is the prolongation operator.

    A V-cycle is used as a preconditioner for GMRES or CG.

    Parameters
    ----------
    A : sp.spmatrix
        The system matrix (N x N). Must be square and sparse.
    num_clusters : int, optional
        Number of coarse-level aggregates. Defaults to max(2, N // 1000).
    p_method : str, optional
        Prolongation method: 'binary' or 'smoothed'. Default is 'smoothed'.
    omega : float, optional
        Jacobi damping factor for prolongation smoothing. Default is 2/3.
    smoothing_passes : int, optional
        Number of Jacobi passes when building smoothed P. Default is 1.
    normalize_P : bool, optional
        Normalize rows of P to sum to 1. Default is True.
    clip_P_negatives : bool, optional
        Clip negative entries in P to 0. Default is False.
    smoother_steps_pre : int, optional
        Number of pre-smoothing Jacobi sweeps per V-cycle. Default is 2.
    smoother_steps_post : int, optional
        Number of post-smoothing Jacobi sweeps per V-cycle. Default is 2.
    smoother_weight : float, optional
        Damping weight for the Jacobi smoother. Default is 1.5.
    coarse_solver : str, optional
        Coarse-level solver: 'direct' (LU) or 'cg'. Default is 'direct'.

    Examples
    --------
    >>> import scipy.sparse as sp
    >>> A = sp.eye(100, format='csr') * 2 - sp.eye(100, k=1, format='csr') - sp.eye(100, k=-1, format='csr')
    >>> b = np.ones(100)
    >>> solver = AMGSolver(A, num_clusters=10)
    >>> result = solver.solve(b)
    >>> print(result)
    """

    def __init__(
        self,
        A: sp.spmatrix,
        num_clusters: Optional[int] = None,
        p_method: str = "smoothed",
        omega: float = 2 / 3,
        smoothing_passes: int = 1,
        normalize_P: bool = True,
        clip_P_negatives: bool = False,
        smoother_steps_pre: int = 2,
        smoother_steps_post: int = 2,
        smoother_weight: float = 1.5,
        coarse_solver: str = "direct",
    ) -> None:
        self._validate_inputs(A)
        self.A = A.tocsr()
        self.N = A.shape[0]
        self.is_symmetric = (self.A != self.A.T).nnz == 0

        self.smoother_steps_pre = smoother_steps_pre
        self.smoother_steps_post = smoother_steps_post
        self.smoother_weight = smoother_weight

        self.n_parts = num_clusters if num_clusters else max(2, self.N // 1000)

        logger.info(
            "Initializing AMGSolver: N=%d, n_parts=%d, p_method='%s', coarse_solver='%s'",
            self.N, self.n_parts, p_method, coarse_solver,
        )

        # Build hierarchy
        self.P = build_prolongation(
            self.A,
            n_parts=self.n_parts,
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        b: np.ndarray,
        method: str = "gmres",
        tol: float = 1e-6,
        max_iter: int = 500,
        callback: Optional[Callable] = None,
    ) -> SolverResult:
        """
        Solve A x = b using the AMG V-cycle as a preconditioner.

        Parameters
        ----------
        b : np.ndarray
            Right-hand side vector of length N.
        method : str, optional
            Outer Krylov method: 'gmres' or 'cg'. Default is 'gmres'.
        tol : float, optional
            Convergence tolerance (relative residual). Default is 1e-6.
        max_iter : int, optional
            Maximum number of Krylov iterations. Default is 500.
        callback : callable, optional
            Optional user callback called at each iteration.
            Signature: callback(iteration, residual, x).

        Returns
        -------
        SolverResult
            Dataclass containing the solution, convergence info, and timing.

        Raises
        ------
        ValueError
            If `b` has wrong shape or an unknown method is specified.
        """
        self._validate_rhs(b)
        if method.lower() not in ("gmres", "cg", "bicgstab", "lgmres"):
            raise ValueError(f"Unknown method: '{method}'. Choose 'gmres', 'cg', 'bicgstab' or 'lgmres'.")

        M_inv = LinearOperator(
            shape=(self.N, self.N),
            matvec=self.v_cycle,
            dtype=self.A.dtype,
        )

        residuals = []
        iteration_counter = [0]

        def _track(xk):
            """Internal callback to record residual history."""
            rel_res = np.linalg.norm(b - self.A @ xk) / (np.linalg.norm(b) + 1e-300)
            residuals.append(rel_res)
            iteration_counter[0] += 1
            if callback is not None:
                callback(iteration=iteration_counter[0], residual=rel_res, x=xk)

        def _track_gmres(rk):
            """GMRES legacy callback receives residual norm directly."""
            residuals.append(float(rk))
            iteration_counter[0] += 1
            if callback is not None:
                callback(iteration=iteration_counter[0], residual=float(rk), x=None)

        logger.info("Starting %s solve (tol=%.1e, max_iter=%d)...", method.upper(), tol, max_iter)
        start_time = time.time()

        if method.lower() == "cg":
            x, info = cg(
                self.A, b, M=M_inv, atol=tol, callback=_track, maxiter=max_iter
            )
        elif method.lower() == "bicgstab":
            x, info = bicgstab(
                self.A, b, M=M_inv, atol=tol, callback=_track, maxiter=max_iter
            )
        elif method.lower() == "lgmres":
            x, info = lgmres(
                self.A, b, M=M_inv, atol=tol, callback=_track, maxiter=max_iter,
                inner_m=30,   # inner GMRES iterations before restart
                outer_k=3,    # number of vectors to carry over between restarts
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
            method=method.lower(),
            tol=tol,
        )

        logger.info(result)
        return result
    
    def solve_standalone(
        self,
        b: np.ndarray,
        tol: float = 1e-6,
        max_iter: int = 500,
        callback: Optional[Callable] = None,
    ) -> SolverResult:
        """
        Solve A x = b using repeated V-cycles only (no Krylov accelerator).
    
        This is the 'pure AMG' approach: the V-cycle is applied directly
        as a solver rather than as a preconditioner. Useful for comparison
        against the preconditioned Krylov methods.
    
        Parameters
        ----------
        b : np.ndarray
            Right-hand side vector.
        tol : float, optional
            Convergence tolerance (relative residual). Default is 1e-6.
        max_iter : int, optional
            Maximum number of V-cycle iterations. Default is 500.
        callback : callable, optional
            Optional user callback. Signature: callback(iteration, residual, x).
    
        Returns
        -------
        SolverResult
            Same dataclass as solve(), for easy comparison.
        """
        self._validate_rhs(b)
    
        u = np.zeros_like(b)
        residuals = []
        b_norm = np.linalg.norm(b) + 1e-300
    
        logger.info("Starting standalone AMG solve (tol=%.1e, max_iter=%d)...", tol, max_iter)
        start_time = time.time()
    
        for i in range(max_iter):
            # Apply V-cycle as a correction
            r = b - self.A @ u
            u = u + self.v_cycle(r)
    
            rel_res = np.linalg.norm(b - self.A @ u) / b_norm
            residuals.append(rel_res)
    
            if callback is not None:
                callback(iteration=i + 1, residual=rel_res, x=u)
    
            if rel_res < tol:
                break
    
        solve_time = time.time() - start_time
        converged = residuals[-1] < tol
    
        result = SolverResult(
            x=u,
            converged=converged,
            iterations=len(residuals),
            residual_history=residuals,
            solve_time=solve_time,
            info=0 if converged else 1,
            method="amg_standalone",
            tol=tol,
        )
    
        logger.info(result)
        return result

    def v_cycle(self, f: np.ndarray) -> np.ndarray:
        """
        Perform a single AMG V-cycle (used as preconditioner action).

        Steps:
          1. Pre-smoothing (weighted Jacobi)
          2. Restrict residual to coarse grid
          3. Solve coarse system
          4. Prolongate correction to fine grid
          5. Post-smoothing (weighted Jacobi)

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

        u = weighted_jacobi(
            self.A, f, u,
            steps=self.smoother_steps_pre,
            weight=self.smoother_weight,
        )

        r_c = self.P.T @ (f - self.A @ u)
        c_c = self._coarse_solver.solve(r_c)
        u += self.P @ c_c

        u = weighted_jacobi(
            self.A, f, u,
            steps=self.smoother_steps_post,
            weight=self.smoother_weight,
        )
        return u

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_inputs(A: sp.spmatrix) -> None:
        """Validate the system matrix at construction time."""
        if not sp.issparse(A):
            raise TypeError(
                f"A must be a scipy sparse matrix, got {type(A).__name__}."
            )
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(
                f"A must be a square 2D matrix, got shape {A.shape}."
            )
        if not np.isfinite(A.data).all():
            raise ValueError("A contains NaN or Inf values.")

    def _validate_rhs(self, b: np.ndarray) -> None:
        """Validate the right-hand side vector before solving."""
        if not isinstance(b, np.ndarray):
            raise TypeError(f"b must be a numpy ndarray, got {type(b).__name__}.")
        if b.shape != (self.N,):
            raise ValueError(
                f"b must have shape ({self.N},), got {b.shape}."
            )
        if not np.isfinite(b).all():
            raise ValueError("b contains NaN or Inf values.")
