# amg-solver &nbsp;·&nbsp; v0.4.0

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-24%20passed-brightgreen)]()

> **Archived:** This repository is a read-only snapshot of the final project version (v0.4.0).
> It is no longer actively maintained.

A modular **two-level Algebraic Multigrid (AMG)** solver in Python, developed at the
**CASE Department, Barcelona Supercomputing Center (BSC)**.

AMG constructs a multilevel hierarchy purely from matrix entries — no geometric
information required — making it directly applicable to irregular industrial FEM
stiffness matrices. The recommended configuration (**binary prolongation + LGMRES**)
achieves residuals **150–550× lower** than a legacy reference solver on seven real
industrial cases (up to N = 2,464,551 DOF), converging in 3 outer iterations (~34 s)
versus 452 s for standalone AMG without convergence.

---

## Solver families

| Class | Strategy | Status |
|-------|----------|--------|
| `AMGSolver` | Classical AMG — binary or Jacobi-smoothed prolongation | ✅ Production |
| `MLAMGSolver` | Smoothed prolongation + MLP-predicted per-cluster ω | 🧪 Experimental |
| `MLProlongationSolver` | MLP-predicted per-node interpolation weights | 🧪 Experimental |
| `GNNProlongationSolver` | GNN-predicted per-node interpolation weights (attention) | 🧪 Experimental |

> **Recommended production configuration:**
> `AMGSolver` with `p_method="binary"` + `method="lgmres"` (inner m=30, outer k=3),
> Jacobi smoother ω=0.7, 2 pre/post sweeps, direct SuperLU coarse solver.

---

## Project structure

```
amg_solver/
├── amg/                          ← importable library
│   ├── __init__.py
│   ├── solver.py                 ← AMGSolver (classical)
│   ├── ml_solver.py              ← MLAMGSolver (ML Jacobi weights)
│   ├── mlp_solver.py             ← MLProlongationSolver (MLP-P)
│   ├── gnn_solver.py             ← GNNProlongationSolver (GNN-P)
│   ├── gnn_model.py              ← GNN architectures (EdgeWeightedSAGE + AttentionGNN)
│   ├── mlp_model.py              ← MLP architecture for ML-P
│   ├── ml_weight_predictor.py    ← MLP architecture for ML-AMG weights
│   ├── prolongation.py           ← P matrix construction (binary / smoothed)
│   ├── smoothers.py              ← Jacobi smoother + coarse solver factory
│   ├── result.py                 ← SolverResult dataclass
│   ├── ml_features.py            ← cluster-level features for ML-AMG
│   ├── mlp_features.py           ← node-level features for MLP-P / GNN-P
│   └── utils.py                  ← I/O helpers, diagnostics, plots
├── benchmarks/
│   ├── run_benchmark.py          ← classical AMG: solver × prolongation comparison
│   ├── run_benchmark_mlp.py      ← classical vs MLP-P
│   ├── run_benchmark_gnn.py      ← classical vs MLP-P vs GNN-P
│   ├── solutionCheck.py          ← validation against reference solution
│   └── amg_weight_predictor.pth  ← pre-trained ML-AMG weight checkpoint
├── tests/
│   ├── conftest.py               ← shared fixtures
│   └── test_solver.py            ← 24-test pytest suite
├── pyproject.toml
└── README.md
```

---

## Installation

```bash
# Core library — classical AMG only (no PyTorch required)
pip install -e .

# With ML/GNN support
pip install -e ".[ml]"

# Full development install (includes pytest)
pip install -e ".[all]"
```

**Requirements:** Python ≥ 3.9, NumPy ≥ 1.24, SciPy ≥ 1.10, pymetis ≥ 2023.1,
pandas ≥ 1.5, matplotlib ≥ 3.6. PyTorch ≥ 2.0 only needed for ML solvers.

---

## Quick start

### Classical AMG (recommended)

```python
import numpy as np
import scipy.sparse as sp
from amg import AMGSolver

# Build a 1D Poisson matrix
n = 1000
A = sp.diags([2*np.ones(n), -np.ones(n-1), -np.ones(n-1)], [0, -1, 1], format="csr")
b = np.ones(n)

# Solve with LGMRES + AMG preconditioner
solver = AMGSolver(A, num_clusters=50, p_method="binary", smoother_weight=0.7)
result = solver.solve(b, method="lgmres", tol=1e-10)
print(result)
# SolverResult(CONVERGED | method=LGMRES | iters=3 | final_residual=4.2e-11 | time=0.034s)
```

### Standalone V-cycle (SPD matrices only)

```python
result = solver.solve_standalone(b, tol=1e-6, max_iter=200)
```

### MLP-Prolongation solver

```python
from amg import MLProlongationSolver

solver = MLProlongationSolver(
    A,
    mlP_model_path="benchmarks/amg_weight_predictor.pth",
    prolongation_method="smoothed",
)
result = solver.solve(b, method="lgmres", tol=1e-10)
```

### GNN-Prolongation solver

```python
from amg import GNNProlongationSolver

solver = GNNProlongationSolver(
    A,
    gnn_model_path="path/to/gnn_model_attn.pth",
    prolongation_method="smoothed",
    inference_batch_size=50_000,    # reduce if memory-constrained
)
result = solver.solve(b, method="lgmres", tol=1e-10)
```

---

## Key parameters — `AMGSolver`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_clusters` | `N // 1000` | Number of coarse-level METIS aggregates |
| `p_method` | `"smoothed"` | Prolongation: `"binary"` or `"smoothed"` |
| `omega` | `2/3` | Jacobi damping factor for P smoothing |
| `smoothing_passes` | `1` | Jacobi passes applied to tentative P |
| `normalize_P` | `True` | Normalize P rows to sum to 1 |
| `clip_P_negatives` | `False` | Clip negative P entries to 0 |
| `smoother_steps_pre` | `2` | Pre-smoothing sweeps per V-cycle |
| `smoother_steps_post` | `2` | Post-smoothing sweeps per V-cycle |
| `smoother_weight` | `1.5` | Damping weight for Jacobi smoother (use 0.7 for standalone) |
| `coarse_solver` | `"direct"` | Coarse-level solver: `"direct"` (SuperLU → LU fallback) or `"cg"` |

---

## Running the tests

```bash
pytest tests/ -v
```

All 24 tests pass. Coverage includes: input validation, prolongation matrix properties
(shape, row sums, clipping), solver convergence on 1D Poisson (CG, LGMRES), standalone
V-cycle, BiCGSTAB interface, callback behaviour, and smoother/coarse-solver factories.

---

## Running the benchmarks

The benchmark scripts read paths from environment variables — no source edits needed:

```bash
# Classical AMG: solver × prolongation comparison
export AMG_BASE_PATH=/path/to/cases
export AMG_CASE=Case4
python benchmarks/run_benchmark.py

# Classical vs MLP-P (multiple cases)
export AMG_CASES=Case1,Case2,Case3
python benchmarks/run_benchmark_mlp.py

# Classical vs MLP-P vs GNN-P
python benchmarks/run_benchmark_gnn.py

# Validate against reference solution
python benchmarks/solutionCheck.py
```

---

## Logging

```python
import logging

# Enable verbose output
logging.basicConfig(level=logging.INFO)

# Silence all AMG output
logging.getLogger("amg").setLevel(logging.WARNING)
```

---

## Background

This library was developed as part of a research project at the
**CASE Department, Barcelona Supercomputing Center (BSC)**, studying AMG methods
applied to large-scale industrial FEM pressure matrices.

Key findings:
- AMG-preconditioned LGMRES is the only configuration that converges reliably across
  all tested PDE types (diffusion-dominated and convection-dominated)
- Binary prolongation + LGMRES achieves a **13× speedup** over standalone AMG with
  better accuracy
- ML-based prolongation (MLP and GNN) did not improve over classical AMG on the
  industrial matrices, due to training distribution mismatch; this is an open
  direction for future work

---

## License

Copyright 2026 Barcelona Supercomputing Center (BSC) — CASE Department.
Licensed under the [Apache License 2.0](LICENSE).
