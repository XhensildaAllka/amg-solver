"""
benchmarks/run_benchmark_mlp.py
=================================
Benchmark comparing Classical AMG vs MLP-Prolongation AMG.

The MLP predicts per-node interpolation strengths, which are used to
build a data-driven prolongation matrix P.

Configuration
-------------
    export AMG_BASE_PATH=/path/to/cases
    export AMG_CASES=Case1,Case2,Case3
    export AMG_MLP_MODEL=/path/to/mlP_model.pth  # optional override

Then run:
    python benchmarks/run_benchmark_mlp.py
"""

import os
import sys
import logging
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from amg import AMGSolver
from amg.mlp_solver import MLProlongationSolver
from amg.utils import load_system, load_reference_solution, print_summary_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_PATH = os.environ.get("AMG_BASE_PATH", "/path/to/your/cases")
_CASES_ENV = os.environ.get("AMG_CASES", "Case1,Case2,Case3")
CASES = [c.strip() for c in _CASES_ENV.split(",")]

_HERE         = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MLP  = os.path.join(_HERE, "..", "training", "models", "mlP_model_suitesparse.pth")
MLP_MODEL_PATH = os.environ.get("AMG_MLP_MODEL", _DEFAULT_MLP)

SMOOTHER_STEPS = 2
COARSE_SOLVER  = "direct"
SOLVER_METHOD  = "lgmres"
TOL            = 1e-10
# ---------------------------------------------------------------------------


def run_case(case_name: str, A, b) -> list:
    results = []

    # Classical AMG
    print(f"\n  [Classical AMG]")
    try:
        r = AMGSolver(
            A, p_method="binary",
            smoother_steps_pre=SMOOTHER_STEPS,
            smoother_steps_post=SMOOTHER_STEPS,
            smoother_weight=0.7,
            coarse_solver=COARSE_SOLVER,
        ).solve(b, method=SOLVER_METHOD, tol=TOL)
        results.append({
            "config": f"{case_name}_classical", "solver": SOLVER_METHOD,
            "time": r.solve_time, "iters": r.iterations,
            "history": r.residual_history, "converged": r.converged, "x": r.x,
        })
        print(f"  {'OK' if r.converged else 'FAIL'} {r}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # MLP-P AMG (smoothed)
    print(f"\n  [MLP-P AMG (smoothed)]")
    if os.path.exists(MLP_MODEL_PATH):
        try:
            r = MLProlongationSolver(
                A, mlP_model_path=MLP_MODEL_PATH,
                prolongation_method="smoothed",
                smoother_steps_pre=SMOOTHER_STEPS,
                smoother_steps_post=SMOOTHER_STEPS,
                smoother_weight=0.7,
                coarse_solver=COARSE_SOLVER,
            ).solve(b, method=SOLVER_METHOD, tol=TOL)
            results.append({
                "config": f"{case_name}_mlp_smoothed", "solver": SOLVER_METHOD,
                "time": r.solve_time, "iters": r.iterations,
                "history": r.residual_history, "converged": r.converged, "x": r.x,
            })
            print(f"  {'OK' if r.converged else 'FAIL'} {r}")
        except Exception as e:
            print(f"  ERROR: {e}")
    else:
        print(f"  SKIP: model not found at {MLP_MODEL_PATH}")

    return results


def main():
    for case in CASES:
        case_path = os.path.join(BASE_PATH, case)
        a_path = os.path.join(case_path, "MATRIXMARKET_MATRIX-1_1.txt")
        b_path = os.path.join(case_path, "MATRIXMARKET_RHS-1_1.txt")
        x_path = os.path.join(case_path, "MATRIXMARKET_UNKNO-1_1.txt")

        print(f"\n{'#'*60}")
        print(f"# CASE: {case}")
        print(f"{'#'*60}")

        try:
            A, b = load_system(a_path, b_path)
            x_ref = load_reference_solution(x_path, A.shape[0])
        except FileNotFoundError as e:
            print(f"  Skipping {case}: {e}")
            continue

        results = run_case(case, A, b)
        if results:
            print_summary_table(results, x_ref=x_ref)


if __name__ == "__main__":
    main()
