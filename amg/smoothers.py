"""
smoothers.py - Pre/post smoothers and coarse-level solvers for AMG.

Contains the weighted Jacobi smoother and wrappers for direct/iterative
coarse-grid solvers.
"""

import logging
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, cg, gmres
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def weighted_jacobi(
    A: sp.csr_matrix,
    f: np.ndarray,
    u: np.ndarray,
    steps: int = 2,
    weight: float = 1.5,
) -> np.ndarray:
    """
    Apply weighted Jacobi iterations to smooth the error.

    Computes: u <- u + weight * D^{-1} * (f - A * u)

    Parameters
    ----------
    A : sp.csr_matrix
        System matrix.
    f : np.ndarray
        Right-hand side vector.
    u : np.ndarray
        Current iterate (modified in place and returned).
    steps : int, optional
        Number of smoothing sweeps. Default is 2.
    weight : float, optional
        Damping weight (omega). Default is 1.5.

    Returns
    -------
    np.ndarray
        Updated iterate after smoothing.
    """
    D = A.diagonal()
    D_inv = np.where(np.abs(D) > 1e-15, 1.0 / D, 0.0)

    for _ in range(steps):
        residual = f - A @ u
        u = u + weight * (D_inv * residual)
    return u


class DirectCoarseSolver:
    """
    Coarse-level solver with automatic fallback chain:

    1. SuperLU direct factorization (default, fastest).
    2. Regularized SuperLU: adds small diagonal perturbation epsilon*I
       if the matrix is exactly singular.
    3. GMRES: used as last resort if both LU attempts fail.
       GMRES handles non-symmetric and near-singular systems robustly.

    The factorization (if successful) is computed once at construction
    time and reused for every V-cycle call.

    Parameters
    ----------
    A_c : sp.csr_matrix
        The coarse-level matrix to factorize.
    epsilon : float, optional
        Regularization parameter added to the diagonal if direct LU
        fails. Default is 1e-10.
    """

    def __init__(self, A_c: sp.csr_matrix, epsilon: float = 1e-10) -> None:
        logger.info("Factorizing coarse matrix (shape=%s, nnz=%d)...", A_c.shape, A_c.nnz)
        self._A_c = A_c
        self._strategy = None

        # Strategy 1: direct LU
        try:
            self._lu = splu(A_c.tocsc())
            self._strategy = 'direct'
            logger.info("Coarse solver: direct LU (success).")
            return
        except RuntimeError as e:
            logger.warning(
                "Direct LU failed (%s). Trying regularized LU...", str(e)
            )

        # Strategy 2: regularized LU (add epsilon * I to diagonal)
        try:
            A_reg = A_c + epsilon * sp.eye(A_c.shape[0], format='csc')
            self._lu = splu(A_reg.tocsc())
            self._strategy = 'regularized'
            logger.warning(
                "Coarse solver: regularized LU (epsilon=%.1e).", epsilon
            )
            return
        except RuntimeError:
            logger.warning(
                "Regularized LU also failed. Falling back to GMRES."
            )

        # Strategy 3: GMRES (most robust fallback)
        self._strategy = 'gmres'
        logger.warning(
            "Coarse solver: GMRES fallback (coarse matrix is singular). "
            "Convergence of the outer solver may be affected."
        )

    def solve(self, r_c: np.ndarray) -> np.ndarray:
        """
        Solve A_c * e_c = r_c using the selected strategy.

        Parameters
        ----------
        r_c : np.ndarray
            Coarse-level residual vector.

        Returns
        -------
        np.ndarray
            Coarse-level correction vector.
        """
        if self._strategy in ('direct', 'regularized'):
            return self._lu.solve(r_c)
        else:
            # GMRES fallback
            c_c, info = gmres(
                self._A_c, r_c, atol=1e-10, rtol=1e-10, maxiter=2000
            )
            if info != 0:
                logger.warning(
                    "Coarse GMRES did not fully converge (info=%d). "
                    "Results may be inaccurate.", info
                )
            return c_c


class IterativeCoarseSolver:
    """
    Coarse-level solver using Conjugate Gradient (CG).

    Useful when the coarse matrix is too large for a direct factorization
    or when memory is constrained.

    Parameters
    ----------
    A_c : sp.csr_matrix
        The coarse-level matrix.
    tol : float, optional
        Convergence tolerance for CG. Default is 1e-12.
    max_iter : int, optional
        Maximum number of CG iterations. Default is 1000.
    """

    def __init__(
        self,
        A_c: sp.csr_matrix,
        tol: float = 1e-12,
        max_iter: int = 1000,
    ) -> None:
        self._A_c = A_c
        self._tol = tol
        self._max_iter = max_iter

    def solve(self, r_c: np.ndarray) -> np.ndarray:
        """
        Solve A_c * e_c = r_c using CG.

        Parameters
        ----------
        r_c : np.ndarray
            Coarse-level residual vector.

        Returns
        -------
        np.ndarray
            Coarse-level correction vector.
        """
        c_c, info = cg(self._A_c, r_c, atol=self._tol, rtol=self._tol, maxiter=self._max_iter)
        if info != 0:
            logger.warning("Coarse CG did not fully converge (info=%d).", info)
        return c_c


def make_coarse_solver(A_c: sp.csr_matrix, method: str = "direct"):
    """
    Factory function to construct the appropriate coarse solver.

    Parameters
    ----------
    A_c : sp.csr_matrix
        Coarse-level matrix.
    method : str, optional
        'direct' for LU factorization, 'cg' for iterative CG.
        Default is 'direct'.

    Returns
    -------
    DirectCoarseSolver or IterativeCoarseSolver

    Raises
    ------
    ValueError
        If an unknown method string is passed.
    """
    if method == "direct":
        return DirectCoarseSolver(A_c)
    elif method == "cg":
        return IterativeCoarseSolver(A_c)
    else:
        raise ValueError(f"Unknown coarse_solver: '{method}'. Choose 'direct' or 'cg'.")