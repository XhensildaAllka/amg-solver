"""
benchmarks/run_benchmark_ml.py
================================
Benchmark comparing classical AMGSolver vs MLAMGSolver across all cases.

Edit ML_MODEL_PATH and CASES below, then run:
    python benchmarks/run_benchmark_ml.py
"""

import os
import sys
import logging
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from amg import AMGSolver
from amg.ml_solver import MLAMGSolver
from amg.utils import load_system, load_reference_solution, print_summary_table, plot_convergence

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# *** USER CONFIGURATION ***
# ---------------------------------------------------------------------------
BASE_PATH = "/home/xallka/Escritorio/BSC/Arya_stifnessMatrix/MatrixSecondtime"

CASES = ["Case1", "Case2", "Case3", "Case4", "Case5", "Case6", "Case7"]

ML_MODEL_PATH = "amg_weight_predictor.pth"

# Shared solver settings
SMOOTHER_STEPS  = 2
COARSE_SOLVER   = "direct"
SOLVER_METHOD   = "lgmres"
TOL             = 1e-10

# ---------------------------------------------------------------------------
# Per-case benchmark
# ---------------------------------------------------------------------------

def run_case(case_name, A, b, x_ref):
    """Run classical vs ML-AMG for a single case. Returns list of result dicts."""
    results = []

    # --- Classical AMG ---
    print(f"\n  [Classical AMG]")
    try:
        classical = AMGSolver(
            A,
            p_method="binary",
            smoother_steps_pre=SMOOTHER_STEPS,
            smoother_steps_post=SMOOTHER_STEPS,
            smoother_weight=0.7,
            coarse_solver=COARSE_SOLVER,
        )
        r = classical.solve(b, method=SOLVER_METHOD, tol=TOL)
        results.append({
            "config": f"{case_name}_classical",
            "solver": SOLVER_METHOD,
            "time": r.solve_time,
            "iters": r.iterations,
            "history": r.residual_history,
            "converged": r.converged,
            "x": r.x,
        })
        print(f"  {'OK' if r.converged else 'FAIL'} {r}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # --- ML-AMG ---
    print(f"\n  [ML-AMG]")
    try:
        ml = MLAMGSolver(
            A,
            ml_model_path=ML_MODEL_PATH,
            use_ml_weights=True,
            p_method="binary",
            smoother_steps_pre=SMOOTHER_STEPS,
            smoother_steps_post=SMOOTHER_STEPS,
            smoother_weight_fallback=0.7,
            coarse_solver=COARSE_SOLVER,
            omega_range=(0.7, 0.95),
        )
        r = ml.solve(b, method=SOLVER_METHOD, tol=TOL)
        results.append({
            "config": f"{case_name}_ml",
            "solver": SOLVER_METHOD,
            "time": r.solve_time,
            "iters": r.iterations,
            "history": r.residual_history,
            "converged": r.converged,
            "x": r.x,
        })
        print(f"  {'OK' if r.converged else 'FAIL'} {r}")
    except Exception as e:
        print(f"  ERROR: {e}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all_cases():
    all_results = []
    all_x_refs  = {}

    for case_name in CASES:
        print(f"\n{'='*70}")
        print(f"  CASE: {case_name}")
        print(f"{'='*70}")

        a_path = os.path.join(BASE_PATH, case_name, "MATRIXMARKET_MATRIX-1_1.txt")
        b_path = os.path.join(BASE_PATH, case_name, "MATRIXMARKET_RHS-1_1.txt")
        x_path = os.path.join(BASE_PATH, case_name, "MATRIXMARKET_UNKNO-1_1.txt")

        # Load system
        try:
            A, b = load_system(a_path, b_path)
            x_ref = load_reference_solution(x_path, A.shape[0])
            all_x_refs[case_name] = x_ref
        except Exception as e:
            print(f"  ERROR loading {case_name}: {e}")
            continue

        # Run benchmark
        case_results = run_case(case_name, A, b, x_ref)
        all_results.extend(case_results)

        # Per-case diagnostics
        print(f"\n  Diagnostics for {case_name}:")
        ref_res = np.linalg.norm(b - A @ x_ref) / np.linalg.norm(b)
        print(f"    x_ref global residual: {ref_res:.4e}")
        for r in case_results:
            if r["x"] is not None:
                true_res = np.linalg.norm(b - A @ r["x"]) / np.linalg.norm(b)
                ref_err  = np.linalg.norm(x_ref - r["x"]) / (np.linalg.norm(x_ref) + 1e-300)
                print(
                    f"    {r['config']:<35} | "
                    f"residual: {true_res:.4e} | "
                    f"vs ref: {ref_err:.4e} | "
                    f"{'CONVERGED' if r['converged'] else 'NOT CONVERGED'}"
                )

    # ---------------------------------------------------------------------------
    # Global summary table
    # ---------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  GLOBAL SUMMARY")
    print(f"{'='*70}")
    print_summary_table(all_results, x_ref=None)

    # ---------------------------------------------------------------------------
    # Pairwise speedup summary
    # ---------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  ML-AMG vs CLASSICAL: SPEEDUP SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Case':<10} | {'Classical (s)':<15} | {'ML-AMG (s)':<12} | {'Speedup':<10} | {'Iter Classical':<16} | {'Iter ML'}")
    print(f"  {'-'*80}")

    for case_name in CASES:
        classical = next(
            (r for r in all_results if r["config"] == f"{case_name}_classical"), None
        )
        ml = next(
            (r for r in all_results if r["config"] == f"{case_name}_ml"), None
        )
        if classical and ml:
            speedup = classical["time"] / ml["time"] if ml["time"] > 0 else float("nan")
            print(
                f"  {case_name:<10} | {classical['time']:<15.2f} | "
                f"{ml['time']:<12.2f} | {speedup:<10.3f} | "
                f"{classical['iters']:<16} | {ml['iters']}"
            )

    # Convergence plot for last case
    if all_results:
        plot_convergence(
            all_results[-2:],
            title=f"Classical vs ML-AMG: {CASES[-1]}",
        )


if __name__ == "__main__":
    run_all_cases()