"""
benchmarks/run_benchmark.py
============================
Benchmark comparing AMG prolongation strategies and Krylov solvers
on a real FEM stiffness matrix.

Figures produced (saved to figures/ subdirectory):
    convergence_krylov.pdf/png       — all Krylov solvers, best prolongation
    convergence_prolongation.pdf/png — all prolongations, GMRES solver
    P_sparsity.pdf/png               — sparsity patterns of P side by side
    standalone_vs_preconditioned.pdf/png — standalone AMG vs LGMRES+AMG

Configuration
-------------
Set environment variables or edit the defaults below:

    export AMG_BASE_PATH=/path/to/case/directory
    export AMG_CASE=Case4
    export AMG_NUM_CLUSTERS=1000

Then run:
    python benchmarks/run_benchmark.py
"""

import os
import sys
import logging

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from amg import AMGSolver
from amg.utils import (
    load_system,
    load_reference_solution,
    plot_convergence,
    plot_convergence_by_solver,
    plot_convergence_by_prolongation,
    plot_prolongation_sparsity_comparison,
    plot_standalone_vs_preconditioned,
    print_summary_table,
    matrix_health_check,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_PATH    = os.environ.get("AMG_BASE_PATH", "/home/xallka/Escritorio/BSC/Arya_stifnessMatrix/MatrixSecondtime")
CASE         = os.environ.get("AMG_CASE",      "Case4")
NUM_CLUSTERS = int(os.environ.get("AMG_NUM_CLUSTERS", "1000"))

CASE_PATH = os.path.join(BASE_PATH, CASE) if CASE else BASE_PATH


A_PATH = os.path.join(CASE_PATH, "MATRIXMARKET_MATRIX-1_1.txt")
B_PATH = os.path.join(CASE_PATH, "MATRIXMARKET_RHS-1_1.txt")
X_PATH = os.path.join(CASE_PATH, "MATRIXMARKET_UNKNO-1_1.txt")

# Output folder for figures (created automatically)
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

SMOOTHER_WEIGHT = 0.7
PRE_POST_SMOOTH = 2

# Prolongation configurations
CONFIGS = [
    {
        "name": "binary",
        "p_method": "binary",
        "omega": 2 / 3,
        "smoothing_passes": 0,
        "normalize_P": False,
        "clip_P_negatives": False,
    },
    {
        "name": "smoothed_1pass_norm",
        "p_method": "smoothed",
        "omega": 2 / 3,
        "smoothing_passes": 1,
        "normalize_P": True,
        "clip_P_negatives": False,
    },
    {
        "name": "smoothed_2pass_clip",
        "p_method": "smoothed",
        "omega": 2 / 3,
        "smoothing_passes": 2,
        "normalize_P": False,
        "clip_P_negatives": True,
    },
]

SOLVERS    = ["cg", "gmres", "bicgstab", "lgmres"]
TOLERANCES = {"cg": 1e-12, "gmres": 1e-10, "bicgstab": 1e-10, "lgmres": 1e-10}

# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark():
    os.makedirs(FIG_DIR, exist_ok=True)

    # 1. Load system
    A, b = load_system(A_PATH, B_PATH)
    x_ref = load_reference_solution(X_PATH, A.shape[0])

    # 2. Health check
    matrix_health_check(A, b)

    results       = []
    prolongations = {}   # name -> P matrix, for sparsity plot

    for cfg in CONFIGS:
        print(f"\n{'='*20} PROLONGATION: {cfg['name'].upper()} {'='*20}")

        amg = AMGSolver(
            A,
            num_clusters=NUM_CLUSTERS,
            p_method=cfg["p_method"],
            omega=cfg["omega"],
            smoothing_passes=cfg["smoothing_passes"],
            normalize_P=cfg["normalize_P"],
            clip_P_negatives=cfg["clip_P_negatives"],
            smoother_steps_pre=PRE_POST_SMOOTH,
            smoother_steps_post=PRE_POST_SMOOTH,
            smoother_weight=SMOOTHER_WEIGHT,
            coarse_solver="direct",
        )

        # Store P for sparsity plot
        prolongations[cfg["name"]] = amg.P

        for s_type in SOLVERS:
            print(f"  Running {s_type.upper()} with config '{cfg['name']}'...")
            tol    = TOLERANCES[s_type]
            result = amg.solve(b, method=s_type, tol=tol, max_iter=500)
            status = "OK" if result.converged else "FAIL"
            print(f"  [{status}] {result}")
            results.append({
                "config":    cfg["name"],
                "solver":    s_type,
                "time":      result.solve_time,
                "iters":     result.iterations,
                "history":   result.residual_history,
                "converged": result.converged,
                "x":         result.x,
            })

    # 3. Summary table
    print_summary_table(results, x_ref=x_ref)

    # ----------------------------------------------------------------
    # 4. Standalone vs preconditioned  (best config: smoothed_1pass_norm)
    # ----------------------------------------------------------------
    print("\nRunning standalone AMG for comparison plot...")
    amg_best = AMGSolver(
        A,
        num_clusters=NUM_CLUSTERS,
        p_method="smoothed",
        omega=2 / 3,
        smoothing_passes=1,
        normalize_P=True,
        smoother_steps_pre=PRE_POST_SMOOTH,
        smoother_steps_post=PRE_POST_SMOOTH,
        smoother_weight=SMOOTHER_WEIGHT,
        coarse_solver="direct",
    )
    result_standalone = amg_best.solve_standalone(b, tol=1e-10, max_iter=500)
    result_lgmres     = next(
        r for r in results
        if r["config"] == "smoothed_1pass_norm" and r["solver"] == "lgmres"
    )

    # Wrap lgmres dict as a lightweight object for the plot function
    class _R:
        pass
    r_pc = _R()
    r_pc.residual_history = result_lgmres["history"]
    r_pc.solve_time       = result_lgmres["time"]
    r_pc.converged        = result_lgmres["converged"]
    r_pc.method           = "lgmres"

    plot_standalone_vs_preconditioned(
        result_standalone, r_pc,
        tol=1e-10,
        save_path=os.path.join(FIG_DIR, "standalone_vs_preconditioned"),
    )

    # ----------------------------------------------------------------
    # 5. Convergence by Krylov solver (smoothed_1pass_norm prolongation)
    # ----------------------------------------------------------------
    plot_convergence_by_solver(
        results,
        prolongation="smoothed_1pass_norm",
        title=(
            "Krylov Solver Convergence — AMG Preconditioner\n"
            "(Smoothed prolongation, 1 pass, normalised, $\\omega = 2/3$, "
            f"{PRE_POST_SMOOTH} pre/post sweeps)"
        ),
        tol=1e-10,
        save_path=os.path.join(FIG_DIR, "convergence_krylov"),
    )

    # ----------------------------------------------------------------
    # 6. Convergence by prolongation strategy (GMRES solver)
    # ----------------------------------------------------------------
    N = A.shape[0]
    plot_convergence_by_prolongation(
        results,
        solver="gmres",
        title=(
            f"Prolongation Strategy Convergence — GMRES Solver\n"
            f"($N = {N:,}$, $\\omega = 2/3$, {PRE_POST_SMOOTH} pre/post sweeps, "
            f"tol $= 10^{{-10}}$)"
        ),
        tol=1e-10,
        save_path=os.path.join(FIG_DIR, "convergence_prolongation"),
    )

    # ----------------------------------------------------------------
    # 7. P sparsity patterns — all prolongation configs side by side
    # ----------------------------------------------------------------
    plot_prolongation_sparsity_comparison(
        prolongations,
        save_path=os.path.join(FIG_DIR, "P_sparsity"),
    )

    print(f"\nAll figures saved to: {os.path.abspath(FIG_DIR)}/")


if __name__ == "__main__":
    run_benchmark()
