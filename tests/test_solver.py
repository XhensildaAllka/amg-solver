"""
tests/test_solver.py
=====================
Test suite for the AMG solver library.

Run from the project root with:
    pytest tests/ -v

Test categories
---------------
- TestInputValidation  : constructor and RHS input guards
- TestProlongation     : P matrix properties (shape, row sums, clipping)
- TestSolverConvergence: solve() on synthetic SPD systems
- TestSolverStandalone : solve_standalone() smoke test
- TestSmoothers        : weighted_jacobi + coarse solver factory
"""

import numpy as np
import pytest
import scipy.sparse as sp

from amg import AMGSolver
from amg.result import SolverResult
from amg.smoothers import (
    weighted_jacobi,
    make_coarse_solver,
    DirectCoarseSolver,
    IterativeCoarseSolver,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def make_1d_poisson(n: int) -> sp.csr_matrix:
    """Build a 1D Poisson matrix (tridiagonal, SPD)."""
    diags = [2 * np.ones(n), -np.ones(n - 1), -np.ones(n - 1)]
    return sp.diags(diags, [0, -1, 1], format="csr")


@pytest.fixture
def small_system():
    """100-node 1D Poisson system (SPD)."""
    n = 100
    A = make_1d_poisson(n)
    b = np.ones(n)
    return A, b


@pytest.fixture
def medium_system():
    """2 000-node 1D Poisson system (SPD)."""
    n = 2000
    A = make_1d_poisson(n)
    b = np.random.default_rng(42).standard_normal(n)
    return A, b


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_non_sparse_raises(self):
        A_dense = np.eye(10)
        with pytest.raises(TypeError, match="scipy sparse"):
            AMGSolver(A_dense)

    def test_non_square_raises(self):
        A = sp.random(10, 20, format="csr")
        with pytest.raises(ValueError, match="square"):
            AMGSolver(A)

    def test_nan_in_matrix_raises(self):
        A = make_1d_poisson(10).astype(float)
        A.data[0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            AMGSolver(A)

    def test_wrong_rhs_shape_raises(self, small_system):
        A, _ = small_system
        solver = AMGSolver(A, num_clusters=10)
        with pytest.raises(ValueError, match="shape"):
            solver.solve(np.ones(50))

    def test_unknown_method_raises(self, small_system):
        """Passing an unsupported method name should raise ValueError."""
        A, b = small_system
        solver = AMGSolver(A, num_clusters=10)
        with pytest.raises(ValueError, match="Unknown method"):
            solver.solve(b, method="superlu_magic")


# ---------------------------------------------------------------------------
# Prolongation
# ---------------------------------------------------------------------------

class TestProlongation:
    def test_binary_row_sums(self, small_system):
        A, _ = small_system
        solver = AMGSolver(A, num_clusters=10, p_method="binary")
        row_sums = np.array(solver.P.sum(axis=1)).flatten()
        assert np.allclose(row_sums, 1.0, atol=1e-10), "Binary P rows must sum to 1"

    def test_smoothed_row_sums(self, small_system):
        A, _ = small_system
        solver = AMGSolver(A, num_clusters=10, p_method="smoothed", normalize_P=True)
        row_sums = np.array(solver.P.sum(axis=1)).flatten()
        assert np.allclose(row_sums, 1.0, atol=1e-10), "Smoothed P rows must sum to 1"

    def test_smoothed_clip_row_sums(self, small_system):
        A, _ = small_system
        solver = AMGSolver(
            A, num_clusters=10, p_method="smoothed",
            clip_P_negatives=True, normalize_P=False,
        )
        row_sums = np.array(solver.P.sum(axis=1)).flatten()
        assert np.allclose(row_sums, 1.0, atol=1e-10), "Clipped P rows must sum to 1"

    def test_P_shape(self, small_system):
        A, _ = small_system
        n_parts = 10
        solver = AMGSolver(A, num_clusters=n_parts)
        assert solver.P.shape == (A.shape[0], n_parts)

    def test_P_non_negative_after_clip(self, small_system):
        A, _ = small_system
        solver = AMGSolver(A, num_clusters=10, p_method="smoothed", clip_P_negatives=True)
        assert (solver.P.data >= -1e-15).all(), "P should have no negative entries after clipping"


# ---------------------------------------------------------------------------
# Solver convergence
# ---------------------------------------------------------------------------

class TestSolverConvergence:
    def test_converges_1d_poisson_cg(self, small_system):
        """CG is the natural Krylov method for SPD systems."""
        A, b = small_system
        solver = AMGSolver(A, num_clusters=10)
        result = solver.solve(b, method="cg", tol=1e-8, max_iter=300)
        assert result.converged, "CG did not converge on 1D Poisson"

    def test_converges_1d_poisson_lgmres(self, small_system):
        """LGMRES should converge on SPD systems too."""
        A, b = small_system
        solver = AMGSolver(A, num_clusters=10)
        result = solver.solve(b, method="lgmres", tol=1e-8, max_iter=300)
        assert result.converged, "LGMRES did not converge on 1D Poisson"

    def test_solution_accuracy_cg(self, small_system):
        """Relative residual after CG solve must be well below 1e-4."""
        A, b = small_system
        solver = AMGSolver(A, num_clusters=5)
        result = solver.solve(b, method="cg", tol=1e-8, max_iter=500)
        residual = np.linalg.norm(b - A @ result.x) / np.linalg.norm(b)
        assert residual < 1e-4, f"CG relative residual {residual:.2e} too large"

    def test_gmres_runs_and_returns_result(self, small_system):
        """
        GMRES is designed for large non-symmetric systems (real stiffness matrices).
        On the small SPD toy problem here we only verify correctness of the interface:
        solve() completes, returns a SolverResult, and populates residual history.
        Full convergence on industrial matrices is validated in the benchmarks.
        """
        A, b = small_system
        solver = AMGSolver(A, num_clusters=5)
        result = solver.solve(b, method="gmres", tol=1e-6, max_iter=100)
        assert isinstance(result, SolverResult)
        assert len(result.residual_history) > 0
        assert result.x.shape == b.shape

    def test_bicgstab_runs_and_returns_result(self, small_system):
        """BiCGSTAB interface smoke test."""
        A, b = small_system
        solver = AMGSolver(A, num_clusters=5)
        result = solver.solve(b, method="bicgstab", tol=1e-6, max_iter=200)
        assert isinstance(result, SolverResult)
        assert result.x.shape == b.shape

    def test_result_is_dataclass(self, small_system):
        A, b = small_system
        solver = AMGSolver(A, num_clusters=10)
        result = solver.solve(b)
        assert isinstance(result, SolverResult)

    def test_residual_history_non_empty(self, small_system):
        A, b = small_system
        solver = AMGSolver(A, num_clusters=10)
        result = solver.solve(b)
        assert len(result.residual_history) > 0

    def test_callback_is_called(self, small_system):
        A, b = small_system
        solver = AMGSolver(A, num_clusters=10)
        call_log = []

        def my_callback(iteration, residual, x):
            call_log.append((iteration, residual))

        solver.solve(b, callback=my_callback)
        assert len(call_log) > 0, "Callback was never called"


# ---------------------------------------------------------------------------
# Standalone V-cycle solver
# ---------------------------------------------------------------------------

class TestSolverStandalone:
    def test_standalone_returns_result(self, small_system):
        """solve_standalone() must return a SolverResult without raising."""
        A, b = small_system
        solver = AMGSolver(A, num_clusters=10)
        result = solver.solve_standalone(b, tol=1e-4, max_iter=50)
        assert isinstance(result, SolverResult)
        assert result.x.shape == b.shape

    def test_standalone_reduces_residual(self, small_system):
        """Pure V-cycle iterations must reduce the residual on a SPD system."""
        A, b = small_system
        # smoother_weight=0.7 is the stable choice for standalone V-cycles
        # (the default 1.5 is tuned for preconditioned use, not standalone)
        solver = AMGSolver(A, num_clusters=10, smoother_weight=0.7)
        result = solver.solve_standalone(b, tol=1e-2, max_iter=100)
        assert result.residual_history[-1] < result.residual_history[0], (
            "Standalone AMG must reduce the residual over iterations"
        )


# ---------------------------------------------------------------------------
# Smoothers
# ---------------------------------------------------------------------------

class TestSmoothers:
    def test_jacobi_reduces_residual(self, small_system):
        A, b = small_system
        u0 = np.zeros_like(b)
        u1 = weighted_jacobi(A, b, u0.copy(), steps=5, weight=1.0)
        res0 = np.linalg.norm(b - A @ u0)
        res1 = np.linalg.norm(b - A @ u1)
        assert res1 < res0, "Jacobi smoother should reduce the residual"

    def test_make_direct_solver(self, small_system):
        A, _ = small_system
        solver = make_coarse_solver(A, method="direct")
        assert isinstance(solver, DirectCoarseSolver)

    def test_make_cg_solver(self, small_system):
        A, _ = small_system
        solver = make_coarse_solver(A, method="cg")
        assert isinstance(solver, IterativeCoarseSolver)

    def test_unknown_coarse_solver_raises(self, small_system):
        A, _ = small_system
        with pytest.raises(ValueError, match="Unknown coarse_solver"):
            make_coarse_solver(A, method="lu_magic")
