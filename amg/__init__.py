"""
amg - Algebraic Multigrid Solver Library
=========================================

A clean, modular two-level Algebraic Multigrid (AMG) solver using
METIS graph partitioning for coarse-level aggregation.

Three solver families are provided:

Classical AMG
-------------
>>> from amg import AMGSolver
>>> solver = AMGSolver(A, p_method="smoothed")
>>> result = solver.solve(b, method="lgmres", tol=1e-10)
>>> print(result)

ML-AMG  (requires PyTorch; learned per-cluster Jacobi weights)
---------------------------------------------------------------
>>> from amg import MLAMGSolver
>>> solver = MLAMGSolver(A, ml_model_path="amg_weight_predictor.pth")
>>> result = solver.solve(b, method="lgmres", tol=1e-10)

MLP-Prolongation  (requires PyTorch; MLP-predicted prolongation)
----------------------------------------------------------------
>>> from amg import MLProlongationSolver
>>> solver = MLProlongationSolver(A, mlP_model_path="mlP_model_suitesparse.pth")
>>> result = solver.solve(b, method="lgmres", tol=1e-10)

GNN-Prolongation  (requires PyTorch; GNN-predicted prolongation)
----------------------------------------------------------------
>>> from amg import GNNProlongationSolver
>>> solver = GNNProlongationSolver(A, gnn_model_path="gnn_model_attn_proxy.pth")
>>> result = solver.solve(b, method="lgmres", tol=1e-10)
"""

from .solver import AMGSolver
from .result import SolverResult

# ML/GNN solvers are optional — require PyTorch
try:
    from .ml_solver import MLAMGSolver
    from .mlp_solver import MLProlongationSolver
    from .gnn_solver import GNNProlongationSolver
    __all__ = [
        "AMGSolver",
        "SolverResult",
        "MLAMGSolver",
        "MLProlongationSolver",
        "GNNProlongationSolver",
    ]
except ImportError:
    __all__ = ["AMGSolver", "SolverResult"]

__version__ = "0.4.0"
__author__ = "Xhensilda Allka — CASE Department, Barcelona Supercomputing Center (BSC)"