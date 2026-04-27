# -*- coding: utf-8 -*-

import sys
import numpy as np
sys.path.insert(0, '/home/xallka/Escritorio/BSC/PythonCode/amg_solver')
from amg import AMGSolver
from amg.utils import load_system, load_reference_solution

BASE_PATH = "/home/xallka/Escritorio/BSC/Arya_stifnessMatrix/MatrixSecondtime/Case5"
A_PATH = BASE_PATH + "/MATRIXMARKET_MATRIX-1_1.txt"
B_PATH = BASE_PATH + "/MATRIXMARKET_RHS-1_1.txt"
X_PATH = BASE_PATH + "/MATRIXMARKET_UNKNO-1_1.txt"

A, b = load_system(A_PATH, B_PATH)
x_ref = load_reference_solution(X_PATH, A.shape[0])

# Solve with AMG
amg = AMGSolver(A, smoother_weight=0.7)
result = amg.solve(b, method='gmres', tol=1e-10)

# ---------------------------------------------------------------------------
def full_diagnostic(label, x, A, b):
    """Print full diagnostic report for solution x."""
    print(f"\n{'='*50}")
    print(f"  SOLUTION: {label}")
    print(f"{'='*50}")

    r = b - A @ x
    diag = A.diagonal()
    b_inf = np.linalg.norm(b, np.inf)

    # 1. Global residual
    global_res = np.linalg.norm(r) / np.linalg.norm(b)
    print(f"Global residual ||b-Ax||/||b||     : {global_res:.4e}")

    # 2. Component-wise residual (safe - avoids division by zero)
    max_rel_safe = np.max(np.abs(r)) / b_inf
    print(f"Max residual / ||b||_inf            : {max_rel_safe:.4e}")

    # 3. Residual distribution
    print(f"Residual percentiles:")
    for p in [50, 75, 90, 95, 99, 100]:
        print(f"  {p:3d}%: {np.percentile(np.abs(r), p):.4e}")

    # 4. Energy norm
    energy = np.sqrt(np.abs(x @ (A @ x)))
    print(f"Energy norm ||x||_A                : {energy:.4e}")

    # 5. Solution statistics
    print(f"Max |displacement|                 : {np.max(np.abs(x)):.4e}")
    print(f"Min |displacement|                 : {np.min(np.abs(x)):.4e}")
    print(f"Mean |displacement|                : {np.mean(np.abs(x)):.4e}")
    print(f"Std |displacement|                 : {np.std(np.abs(x)):.4e}")

    # 6. Worst 10 equations
    worst_indices = np.argsort(np.abs(r))[-10:]
    print(f"\nWorst 10 equations:")
    print(f"  {'Index':<12} {'Residual':<15} {'Diagonal':<15} {'b value':<15} {'x value':<15}")
    print(f"  {'-'*70}")
    for idx in worst_indices[::-1]:
        print(f"  {idx:<12} {r[idx]:<15.4e} {diag[idx]:<15.4e} "
              f"{b[idx]:<15.4e} {x[idx]:<15.4e}")

    # 7. Residual consistency check
    # Are errors random (good) or systematic (bad)?
    r_nonzero_b = r[np.abs(b) > 1e-15]
    if len(r_nonzero_b) > 0:
        mean_r = np.mean(r_nonzero_b)
        std_r = np.std(r_nonzero_b)
        print(f"\nResidual on loaded DOFs (b != 0):")
        print(f"  Mean residual (systematic error): {mean_r:.4e}")
        print(f"  Std  residual (random error)    : {std_r:.4e}")
        print(f"  Ratio mean/std                  : {abs(mean_r)/std_r:.4f}")
        print(f"  (ratio << 1 means errors are random = good)")

# ---------------------------------------------------------------------------
# Run diagnostics for both solutions
full_diagnostic("AMG (our solution)", result.x, A, b)
full_diagnostic("Reference solution (x_ref)", x_ref, A, b)

# ---------------------------------------------------------------------------
# Direct comparison
print(f"\n{'='*50}")
print(f" DIRECT COMPARISON")
print(f"{'='*50}")
r_ours = b - A @ result.x
r_ref  = b - A @ x_ref

print(f"Global residual  - Ours  : {np.linalg.norm(r_ours)/np.linalg.norm(b):.4e}")
print(f"Global residual  - x_ref : {np.linalg.norm(r_ref)/np.linalg.norm(b):.4e}")
print(f"Energy norm      - Ours  : {np.sqrt(np.abs(result.x @ (A @ result.x))):.4e}")
print(f"Energy norm      - x_ref : {np.sqrt(np.abs(x_ref @ (A @ x_ref))):.4e}")
print(f"Max |residual|   - Ours  : {np.max(np.abs(r_ours)):.4e}")
print(f"Max |residual|   - x_ref : {np.max(np.abs(r_ref)):.4e}")
print(f"Diff between solutions   : {np.linalg.norm(result.x - x_ref)/np.linalg.norm(x_ref):.4e}")