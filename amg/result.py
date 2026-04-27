"""
result.py - SolverResult dataclass for AMG solver outputs.
"""

from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class SolverResult:
    """
    Container for the output of an AMG solve.

    Attributes
    ----------
    x : np.ndarray
        The approximate solution vector.
    converged : bool
        True if the solver reached the requested tolerance.
    iterations : int
        Number of iterations performed.
    residual_history : List[float]
        Relative residual norm at each iteration.
    solve_time : float
        Wall-clock time in seconds for the solve phase.
    info : int
        Raw info code from scipy (0 = success, >0 = max iters reached,
        <0 = breakdown).
    method : str
        Iterative method used ('cg' or 'gmres').
    tol : float
        Tolerance requested by the user.

    Examples
    --------
    >>> result = solver.solve(b)
    >>> if result.converged:
    ...     print(f"Solved in {result.iterations} iterations")
    ... else:
    ...     print("Did not converge!")
    """
    x: np.ndarray
    converged: bool
    iterations: int
    residual_history: List[float]
    solve_time: float
    info: int
    method: str
    tol: float

    def __repr__(self) -> str:
        status = "CONVERGED" if self.converged else "NOT CONVERGED"
        final = self.residual_history[-1] if self.residual_history else float("nan")
        return (
            f"SolverResult({status} | method={self.method.upper()} | "
            f"iters={self.iterations} | final_residual={final:.2e} | "
            f"time={self.solve_time:.4f}s)"
        )
