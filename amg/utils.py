"""
utils.py - Utility functions for the AMG solver library.

Includes matrix health diagnostics, Matrix Market file loaders,
and convergence plot helpers.
"""

import logging
import numpy as np
import scipy.sparse as sp
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Matrix diagnostics
# ---------------------------------------------------------------------------

def matrix_health_check(A: sp.csr_matrix, b: np.ndarray) -> None:
    """
    Print a diagnostic report for the linear system (A, b).

    Checks include sparsity, diagonal dominance, condition number estimate,
    and RHS scaling.

    Parameters
    ----------
    A : sp.csr_matrix
        System matrix.
    b : np.ndarray
        Right-hand side vector.
    """
    N = A.shape[0]
    sparsity = (1 - A.nnz / (N ** 2)) * 100

    logger.info("=" * 30)
    logger.info("--- Matrix Health Report ---")
    logger.info("=" * 30)
    logger.info("Nodes (N)         : %d", N)
    logger.info("Non-zeros (nnz)   : %d", A.nnz)
    logger.info("Sparsity          : %.4f%%", sparsity)
    logger.info("Symmetric         : %s", (A != A.T).nnz == 0)

    diag = A.diagonal()
    num_zeros = np.sum(np.abs(diag) < 1e-15)
    row_sums = np.array(np.abs(A).sum(axis=1)).flatten()
    off_diag_sums = row_sums - np.abs(diag)
    is_dominant = bool(np.all(np.abs(diag) >= off_diag_sums))

    logger.info("Zero Diagonals    : %d", num_zeros)
    logger.info("Diag Dominant     : %s", is_dominant)
    logger.info("Diag Range        : [%.2e, %.2e]", diag.min(), diag.max())

    b_norm = np.linalg.norm(b)
    logger.info("||b||             : %.4e", b_norm)
    if b_norm > 0 and np.abs(diag).mean() > 0:
        logger.info("Relative RHS Scale: %.2e", b_norm / np.abs(diag).mean())

    logger.info("-" * 30)


# ---------------------------------------------------------------------------
# Matrix Market loaders
# ---------------------------------------------------------------------------

def load_matrix_market(filepath: str) -> pd.DataFrame:
    """
    Load a Matrix Market file into a DataFrame, skipping comment lines.

    Parameters
    ----------
    filepath : str
        Path to the .txt or .mtx Matrix Market file.

    Returns
    -------
    pd.DataFrame
        DataFrame with the numeric data (header row removed).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    import os
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath, sep=r"\s+", comment="%", header=None)
    return df.drop(df.index[0]).reset_index(drop=True)


def load_system(a_path: str, b_path: str):
    """
    Load a sparse linear system (A, b) from Matrix Market files.

    The matrix dimensions are inferred from the maximum row/column indices
    in the data (1-based indexing is converted to 0-based).

    Parameters
    ----------
    a_path : str
        Path to the Matrix Market file for matrix A.
    b_path : str
        Path to the Matrix Market file for RHS vector b.

    Returns
    -------
    A : sp.csr_matrix
        The system matrix.
    b : np.ndarray
        The right-hand side vector.
    """
    logger.info("Loading matrix from: %s", a_path)
    df_mat = load_matrix_market(a_path)

    rows = df_mat[0].astype(int).max()
    cols = df_mat[1].astype(int).max()
    N = max(rows, cols)

    A = sp.coo_matrix(
        (df_mat[2], (df_mat[0].astype(int) - 1, df_mat[1].astype(int) - 1)),
        shape=(N, N),
    ).tocsr()

    logger.info("Loading RHS from  : %s", b_path)
    df_rhs = load_matrix_market(b_path)
    b = np.zeros(N)

    if df_rhs.shape[1] > 1:
        indices = df_rhs[0].values.astype(int) - 1
        b[indices] = df_rhs[1].values
    else:
        b_values = df_rhs[0].values
        b[: len(b_values)] = b_values

    logger.info("System loaded: A shape=%s, nnz=%d, ||b||=%.4e", A.shape, A.nnz, np.linalg.norm(b))
    return A, b


# ---------------------------------------------------------------------------
# Plot style — applied once at import time
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "text.usetex":        False,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    9.5,
    "legend.framealpha":  0.92,
    "legend.edgecolor":   "#cccccc",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.linestyle":     "--",
    "grid.alpha":         0.45,
    "grid.linewidth":     0.6,
    "lines.linewidth":    1.8,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.08,
})

# Colour-blind-friendly palette (Okabe-Ito)
_COLORS = ["#0072B2", "#D55E00", "#009E73", "#E69F00",
           "#56B4E9", "#CC79A7", "#F0E442", "#000000"]
_LINES  = ["-", "--", "-.", ":", (0,(3,1,1,1)), (0,(5,1))]
_MARKS  = ["o", "s", "D", "^", "v", "P"]


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_convergence(
    results: List[Dict],
    title: str = "AMG Convergence Comparison",
    save_path: Optional[str] = None,
    tol: Optional[float] = None,
) -> None:
    """
    Plot residual convergence histories for multiple solver configurations.

    Produces a publication-quality semi-log plot saved as both PDF and PNG
    (if save_path is provided) or displayed interactively.

    Parameters
    ----------
    results : list of dict
        Each dict must contain: 'config' (str), 'solver' (str),
        'time' (float), 'iters' (int), 'history' (list[float]),
        'converged' (bool).
    title : str, optional
        Figure title.
    save_path : str, optional
        Base file path without extension.  Both <save_path>.pdf and
        <save_path>.png are written.  If None the figure is shown
        interactively.
    tol : float, optional
        If given, a horizontal tolerance line is drawn.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    for idx, r in enumerate(results):
        if not r.get("history"):
            continue
        color  = _COLORS[idx % len(_COLORS)]
        ls     = _LINES[idx % len(_LINES)]
        status = "✓" if r.get("converged") else "✗"
        label  = (
            f"{r['config']} + {r['solver'].upper()}  "
            f"({r['iters']} iter, {r['time']:.1f} s) {status}"
        )
        ax.semilogy(r["history"], color=color, linestyle=ls,
                    linewidth=1.9, label=label, zorder=3)

    if tol is not None:
        ax.axhline(tol, color="#888888", linewidth=0.9,
                   linestyle=":", zorder=1)
        ax.text(0.5, tol * 1.25,
                f"tol = {tol:.0e}",
                fontsize=8.5, color="#666666", ha="left")

    ax.set_xlabel("Outer Iteration")
    ax.set_ylabel(r"Relative Residual $\|\mathbf{r}_k\| / \|\mathbf{b}\|$")
    ax.set_title(title, pad=9)
    ax.set_xlim(left=0)
    ax.legend(loc="upper right", ncol=1)
    fig.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}.pdf")
        fig.savefig(f"{save_path}.png")
        logger.info("Convergence plot saved to: %s.pdf / .png", save_path)
        plt.close(fig)
    else:
        plt.show()


def plot_convergence_by_solver(
    results: List[Dict],
    prolongation: str,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    tol: Optional[float] = None,
) -> None:
    """
    Plot convergence histories for all Krylov solvers sharing the same
    prolongation strategy.  One curve per solver.

    Parameters
    ----------
    results : list of dict
        Full benchmark result list (all configs × solvers).
    prolongation : str
        The prolongation config name to filter on (e.g. 'smoothed_1pass_norm').
    title : str, optional
        Override the auto-generated title.
    save_path : str, optional
        Base path (no extension).  PDF + PNG saved if provided.
    tol : float, optional
        Tolerance line.
    """
    subset = [r for r in results if r["config"] == prolongation and r.get("history")]
    if not subset:
        logger.warning("No results found for prolongation='%s'.", prolongation)
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    for idx, r in enumerate(subset):
        color  = _COLORS[idx % len(_COLORS)]
        ls     = _LINES[idx % len(_LINES)]
        # LGMRES: add markers to highlight the few outer iterations
        use_markers = r["solver"].lower() == "lgmres"
        label = (
            f"{r['solver'].upper()}  "
            f"({r['iters']} iter, {r['time']:.1f} s)"
            f"{'  ✓' if r.get('converged') else '  ✗'}"
        )
        kwargs = dict(color=color, linestyle=ls, linewidth=1.9,
                      label=label, zorder=3)
        if use_markers:
            kwargs.update(marker=_MARKS[idx % len(_MARKS)],
                          markersize=7, linewidth=2.2)
        ax.semilogy(r["history"], **kwargs)

    if tol is not None:
        ax.axhline(tol, color="#888888", linewidth=0.9, linestyle=":", zorder=1)
        ax.text(0.5, tol * 1.25, f"tol = {tol:.0e}",
                fontsize=8.5, color="#666666", ha="left")

    cfg_label = prolongation.replace("_", "\\_") if title is None else ""
    auto_title = (
        f"Krylov Solver Convergence — AMG Preconditioner\n"
        f"(Prolongation: {cfg_label}, $\\omega = 2/3$, 2 pre/post sweeps)"
    )
    ax.set_xlabel("Outer Iteration")
    ax.set_ylabel(r"Relative Residual $\|\mathbf{r}_k\| / \|\mathbf{b}\|$")
    ax.set_title(title if title else auto_title, pad=9)
    ax.set_xlim(left=0)
    ax.legend(loc="upper right")
    fig.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}.pdf")
        fig.savefig(f"{save_path}.png")
        logger.info("Solver convergence plot saved to: %s.pdf / .png", save_path)
        plt.close(fig)
    else:
        plt.show()


def plot_convergence_by_prolongation(
    results: List[Dict],
    solver: str,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    tol: Optional[float] = None,
) -> None:
    """
    Plot convergence histories for all prolongation strategies sharing the
    same outer Krylov solver.  One curve per prolongation config.

    Parameters
    ----------
    results : list of dict
        Full benchmark result list.
    solver : str
        Krylov solver to filter on (e.g. 'gmres').
    title : str, optional
        Override the auto-generated title.
    save_path : str, optional
        Base path (no extension).  PDF + PNG saved if provided.
    tol : float, optional
        Tolerance line.
    """
    subset = [r for r in results
              if r["solver"].lower() == solver.lower() and r.get("history")]
    if not subset:
        logger.warning("No results found for solver='%s'.", solver)
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    for idx, r in enumerate(subset):
        color = _COLORS[idx % len(_COLORS)]
        ls    = _LINES[idx % len(_LINES)]
        label = (
            f"{r['config']}  "
            f"({r['iters']} iter, {r['time']:.1f} s)"
            f"{'  ✓' if r.get('converged') else '  ✗'}"
        )
        ax.semilogy(r["history"], color=color, linestyle=ls,
                    linewidth=1.9, label=label, zorder=3)

    if tol is not None:
        ax.axhline(tol, color="#888888", linewidth=0.9, linestyle=":", zorder=1)
        ax.text(0.5, tol * 1.25, f"tol = {tol:.0e}",
                fontsize=8.5, color="#666666", ha="left")

    auto_title = (
        f"Prolongation Strategy Convergence — {solver.upper()} Solver\n"
        f"($N = {{}},$ $\\omega = 2/3$, 2 pre/post sweeps)"
    )
    ax.set_xlabel("Outer Iteration")
    ax.set_ylabel(r"Relative Residual $\|\mathbf{r}_k\| / \|\mathbf{b}\|$")
    ax.set_title(title if title else auto_title.format(""), pad=9)
    ax.set_xlim(left=0)
    ax.legend(loc="upper right")
    fig.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}.pdf")
        fig.savefig(f"{save_path}.png")
        logger.info("Prolongation convergence plot saved to: %s.pdf / .png", save_path)
        plt.close(fig)
    else:
        plt.show()


def plot_standalone_vs_preconditioned(
    result_standalone,
    result_precond,
    tol: Optional[float] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Compare standalone AMG V-cycles against AMG-preconditioned Krylov.

    Single plot with dual x-axes:
      - Bottom x-axis: V-cycle count for standalone AMG (0 → max_iter)
      - Top x-axis:    outer iteration count for preconditioned Krylov (0 → n_pc)
    Both curves share the same y-axis (relative residual), so the residual
    level reached by LGMRES in 6 iterations is directly comparable to
    where standalone AMG is still stuck after 500 iterations.

    Parameters
    ----------
    result_standalone : SolverResult
        Output of solver.solve_standalone().
    result_precond : SolverResult
        Output of solver.solve() with a Krylov method.
    tol : float, optional
        Tolerance line.
    save_path : str, optional
        Base path (no extension).  PDF + PNG saved if provided.
    """
    n_sa = len(result_standalone.residual_history)
    n_pc = len(result_precond.residual_history)
    method_label = result_precond.method.upper().replace("_", "+")

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(12.0, 5.0),
        gridspec_kw={"wspace": 0.35},
    )

    # Compute shared y-axis limits across both curves
    all_vals = (
        [v for v in result_standalone.residual_history if v > 0] +
        [v for v in result_precond.residual_history if v > 0]
    )
    y_min = min(all_vals) * 0.1
    y_max = max(all_vals) * 5
    if tol is not None:
        y_min = min(y_min, tol * 0.05)

    # --- Left: standalone AMG ---
    status_sa = "Did Not Converge (DNF)" if not result_standalone.converged else "Converged"
    ax_left.semilogy(
        result_standalone.residual_history,
        color=_COLORS[1], linewidth=1.7, zorder=3,
    )
    if tol is not None:
        ax_left.axhline(tol, color="#888888", linewidth=0.9, linestyle=":", zorder=1)
        ax_left.text(1, tol * 1.4, f"tol = {tol:.0e}", fontsize=8.5, color="#666666")
    ax_left.set_title(
        f"Standalone AMG V-cycles\n"
        f"{n_sa} iterations  |  {result_standalone.solve_time:.1f} s  |  {status_sa}",
        fontsize=10.5, pad=7,
    )
    ax_left.set_xlabel("V-cycle Iteration")
    ax_left.set_ylabel(r"Relative Residual $\|\mathbf{r}_k\| / \|\mathbf{b}\|$")
    ax_left.set_xlim(0, n_sa)
    ax_left.set_ylim(y_min, y_max)

    # --- Right: preconditioned Krylov ---
    status_pc = "Converged" if result_precond.converged else "Did Not Converge"
    ax_right.semilogy(
        result_precond.residual_history,
        color=_COLORS[0], linewidth=2.2, linestyle="--",
        marker=_MARKS[1], markersize=9, zorder=4,
    )
    if tol is not None:
        ax_right.axhline(tol, color="#888888", linewidth=0.9, linestyle=":", zorder=1)
        ax_right.text(0.05, tol * 1.4, f"tol = {tol:.0e}", fontsize=8.5, color="#666666")
    ax_right.set_title(
        f"{method_label} + AMG Preconditioner\n"
        f"{n_pc} iterations  |  {result_precond.solve_time:.1f} s  |  {status_pc}",
        fontsize=10.5, pad=7,
    )
    ax_right.set_xlabel("Outer Iteration")
    ax_right.set_ylabel(r"Relative Residual $\|\mathbf{r}_k\| / \|\mathbf{b}\|$")
    ax_right.set_xlim(0, n_pc - 1)
    ax_right.set_ylim(y_min, y_max)   # same y range as left panel
    ax_right.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    fig.suptitle(
        "Standalone AMG vs. AMG-Preconditioned Krylov\n"
        "(Smoothed prolongation, 1 pass, normalised, $\\omega = 2/3$)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}.pdf")
        fig.savefig(f"{save_path}.png")
        logger.info("Standalone vs preconditioned plot saved to: %s.pdf / .png", save_path)
        plt.close(fig)
    else:
        plt.show()

def load_reference_solution(filepath: str, N: int) -> np.ndarray:
    """
    Load a reference solution vector from a Matrix Market file.

    Parameters
    ----------
    filepath : str
        Path to the Matrix Market file containing the reference solution.
    N : int
        Expected size of the solution vector.

    Returns
    -------
    np.ndarray
        Reference solution vector of length N.
    """
    logger.info("Loading reference solution from: %s", filepath)
    df = load_matrix_market(filepath)
    x_ref = np.zeros(N)

    if df.shape[1] > 1:
        indices = df[0].values.astype(int) - 1
        x_ref[indices] = df[1].values
    else:
        x_ref[:len(df[0].values)] = df[0].values

    logger.info("Reference solution loaded: ||x_ref||=%.4e", np.linalg.norm(x_ref))
    return x_ref


def plot_prolongation_sparsity(P: sp.csr_matrix, title: str = "P sparsity",
                               save_path: Optional[str] = None) -> None:
    """
    Visualize the sparsity pattern of a single prolongation matrix P.

    Parameters
    ----------
    P : sp.csr_matrix
        Prolongation matrix (N x n_c).
    title : str, optional
        Plot title.
    save_path : str, optional
        Base path (no extension).  PDF + PNG saved if provided.
    """
    nnz_per_row = np.diff(P.indptr)
    mean_nnz    = nnz_per_row.mean()

    fig, ax = plt.subplots(figsize=(5.5, 7.5))
    ax.spy(P, markersize=0.8, aspect="auto", color="#1f4e79", origin="upper")
    ax.set_xlabel(f"Coarse DOF index  ($n_c = {P.shape[1]:,}$)", fontsize=10)
    ax.set_ylabel(f"Fine DOF index  ($N = {P.shape[0]:,}$)",    fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.text(0.02, 0.97,
            f"nnz = {P.nnz:,}\nMean row nnz = {mean_nnz:.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", alpha=0.88))
    fig.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}.pdf")
        fig.savefig(f"{save_path}.png")
        logger.info("Sparsity plot saved to: %s.pdf / .png", save_path)
        plt.close(fig)
    else:
        plt.show()


def plot_prolongation_sparsity_comparison(
    prolongation_dict: Dict[str, sp.csr_matrix],
    save_path: Optional[str] = None,
) -> None:
    """
    Plot sparsity patterns of multiple prolongation matrices side by side.

    Parameters
    ----------
    prolongation_dict : dict {label: P_matrix}
        Ordered dict mapping a short config name to its P matrix.
        E.g. {'binary': P_bin, 'smoothed_1pass_norm': P_smo}
    save_path : str, optional
        Base path (no extension).  PDF + PNG saved if provided.
    """
    configs = list(prolongation_dict.items())
    n = len(configs)
    colors = ["#1f4e79", "#1a5e20", "#7b1f3a", "#4a3000"]

    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 7.5))
    if n == 1:
        axes = [axes]

    for ax, (label, P), col in zip(axes, configs, colors):
        nnz_per_row = np.diff(P.indptr)
        mean_nnz    = nnz_per_row.mean()

        ax.spy(P, markersize=0.8, aspect="auto", color=col, origin="upper")
        ax.set_xlabel(f"Coarse DOF index  ($n_c = {P.shape[1]:,}$)", fontsize=9.5)
        ax.set_ylabel(f"Fine DOF index  ($N = {P.shape[0]:,}$)",     fontsize=9.5)
        ax.set_title(label.replace("_", " ").title(), fontsize=10.5)
        ax.text(0.02, 0.97,
                f"nnz = {P.nnz:,}\nMean row nnz = {mean_nnz:.2f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#cccccc", alpha=0.88))

    fig.suptitle(
        "Sparsity Patterns of Prolongation Matrix $\\mathbf{P}$  ($N \\times n_c$)\n"
        "Block structure reflects METIS graph partitioning",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}.pdf")
        fig.savefig(f"{save_path}.png")
        logger.info("P sparsity comparison saved to: %s.pdf / .png", save_path)
        plt.close(fig)
    else:
        plt.show()


def print_summary_table(results: List[Dict], x_ref: np.ndarray = None) -> None:
    """
    Print a formatted summary table of benchmark results.

    Parameters
    ----------
    results : list of dict
        Each dict must have keys: 'config', 'solver', 'time', 'iters', 'history'.
    x_ref : np.ndarray, optional
        Reference solution vector. If provided, relative error is computed.
    """
    sep = "=" * 100
    if x_ref is not None:
        header = f"{'Prolongation':<25} | {'Solver':<10} | {'Time (s)':<10} | {'Iter':<6} | {'Final Resid':<15} | {'Rel Error':<15}"
        print(f"\n{sep}")
        print(header)
        print("-" * 100)
        for r in results:
            final_resid = r["history"][-1] if r["history"] else float("nan")
            # compute true relative error
            x_ref_norm = np.linalg.norm(x_ref)
            rel_err = np.linalg.norm(x_ref - r["x"]) / x_ref_norm if x_ref_norm > 0 else float("nan")
            print(
                f"{r['config']:<25} | {r['solver'].upper():<10} | "
                f"{r['time']:<10.4f} | {r['iters']:<6} | "
                f"{final_resid:<15.4e} | {rel_err:<15.4e}"
            )
    else:
        header = f"{'Prolongation':<25} | {'Solver':<10} | {'Time (s)':<10} | {'Iter':<6} | {'Final Resid':<15}"
        print(f"\n{sep}")
        print(header)
        print("-" * 100)
        for r in results:
            final_resid = r["history"][-1] if r["history"] else float("nan")
            print(
                f"{r['config']:<25} | {r['solver'].upper():<10} | "
                f"{r['time']:<10.4f} | {r['iters']:<6} | {final_resid:<15.4e}"
            )
    print(sep)