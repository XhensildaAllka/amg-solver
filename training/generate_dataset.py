"""
training/generate_dataset.py
==============================
Unified dataset generator for ML-P (MLP) and GNN-P training.

Scans a local folder of SuiteSparse .mtx files (the same folder used
by mlP_dataset_generator_suitesparse.py) and produces:

  --mode mlp   → flat (features, target) pairs  → feed into train_mlp.py
  --mode gnn   → graph dicts with edge structure → feed into train_gnn.py

Target options:
  --target proxy       geometric hardness score (fast, original approach)
  --target convergence actual V-cycle error reduction per node (slower, better)

The 10 node features are identical in both modes — same as mlP_feature_extractor.py:
    0  diagonal A_ii
    1  row sum |A_i*|
    2  degree
    3  max |A_ij|
    4  mean |A_ij|
    5  cluster size
    6  same-cluster neighbor count
    7  different-cluster neighbor count
    8  mean diagonal of neighbors
    9  mean row sum of neighbors

Usage
-----
    # Generate GNN dataset with convergence targets (recommended for GNN)
    python training/generate_dataset.py \\
        --matrix_dir suitesparse_matrices \\
        --output gnn_dataset_conv.pkl \\
        --mode gnn --target convergence

    # Generate MLP dataset with proxy targets (matches your existing training)
    python training/generate_dataset.py \\
        --matrix_dir suitesparse_matrices \\
        --output mlp_dataset.pkl \\
        --mode mlp --target proxy

    # Merge multiple checkpoint datasets
    python training/generate_dataset.py --merge \\
        --inputs gnn_dataset_conv_ckpt_5.pkl gnn_dataset_conv_ckpt_10.pkl \\
        --output gnn_dataset_merged.pkl
"""

import os
import sys
import pickle
import time
import argparse
import numpy as np
import scipy.sparse as sp
import scipy.io
import pymetis

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# FEATURE EXTRACTION
# Matches mlP_feature_extractor.py exactly — 10 features per node
# ============================================================

def extract_node_features(A: sp.csr_matrix, membership: np.ndarray, num_clusters: int) -> np.ndarray:
    """
    Extract 10 AMG-relevant node features.
    Identical to mlP_feature_extractor.extract_mlP_features() but vectorized.
    """
    N     = A.shape[0]
    diag  = A.diagonal()
    absA  = abs(A)

    row_sum  = np.array(absA.sum(axis=1)).flatten()
    degree   = np.diff(A.indptr)

    # max / mean per row (small loop only over non-empty rows)
    max_abs  = np.zeros(N)
    mean_abs = np.zeros(N)
    for i in np.where(degree > 0)[0]:
        s, e = A.indptr[i], A.indptr[i+1]
        v = np.abs(A.data[s:e])
        max_abs[i]  = v.max()
        mean_abs[i] = v.mean()

    # cluster size
    cluster_size = np.zeros(N)
    for c in range(num_clusters):
        idx = np.where(membership == c)[0]
        cluster_size[idx] = len(idx)

    # neighbor counts and mean neighbor stats (vectorized via COO)
    A_coo   = A.tocoo()
    row_idx = A_coo.row
    col_idx = A_coo.col

    same_mask = membership[row_idx] == membership[col_idx]
    neigh_same  = np.zeros(N)
    neigh_other = np.zeros(N)
    np.add.at(neigh_same,  row_idx[same_mask],  1)
    np.add.at(neigh_other, row_idx[~same_mask], 1)

    mean_neigh_diag   = np.zeros(N)
    mean_neigh_rowsum = np.zeros(N)
    np.add.at(mean_neigh_diag,   row_idx, diag[col_idx])
    np.add.at(mean_neigh_rowsum, row_idx, row_sum[col_idx])
    safe_deg = np.maximum(degree, 1)
    mean_neigh_diag   /= safe_deg
    mean_neigh_rowsum /= safe_deg
    isolated = degree == 0
    mean_neigh_diag[isolated]   = diag[isolated]
    mean_neigh_rowsum[isolated] = row_sum[isolated]

    return np.column_stack([
        diag, row_sum, degree, max_abs, mean_abs,
        cluster_size, neigh_same, neigh_other,
        mean_neigh_diag, mean_neigh_rowsum,
    ]).astype(np.float32)


# ============================================================
# TARGET COMPUTATION
# ============================================================

def build_classical_P(A: sp.csr_matrix, membership: np.ndarray, num_clusters: int) -> sp.csr_matrix:
    """Standard one-sweep smoothed aggregation prolongation."""
    N = A.shape[0]
    P_tent = sp.csr_matrix(
        (np.ones(N), (np.arange(N), membership)),
        shape=(N, num_clusters),
    )
    D_inv = sp.diags(1.0 / (A.diagonal() + 1e-12))
    return P_tent - (2.0 / 3.0) * (D_inv @ (A @ P_tent))


def proxy_targets(A: sp.csr_matrix, P: sp.csr_matrix, membership: np.ndarray) -> np.ndarray:
    """
    Geometric hardness score — fast, matches original mlP_dataset_generator_suitesparse.py.
    hardness = (off_ratio + (1 - diag_dom) + (1 - same_frac)) / 3
    """
    N    = A.shape[0]
    diag = np.abs(A.diagonal())

    # off/own weights from P (vectorized)
    P_coo    = P.tocoo()
    p_rows   = P_coo.row
    p_cols   = P_coo.col
    p_data   = np.abs(P_coo.data)
    own_mask = p_cols == membership[p_rows]
    off_w    = np.zeros(N)
    own_w    = np.zeros(N)
    np.add.at(off_w, p_rows[~own_mask], p_data[~own_mask])
    np.add.at(own_w, p_rows[own_mask],  p_data[own_mask])
    off_ratio = off_w / (own_w + off_w + 1e-12)

    row_sum_off = np.array(np.abs(A).sum(axis=1)).ravel() - diag
    diag_dom    = diag / (diag + row_sum_off + 1e-12)

    A_coo    = A.tocoo()
    row_idx  = A_coo.row
    col_idx  = A_coo.col
    same_m   = membership[row_idx] == membership[col_idx]
    same_cnt = np.zeros(N)
    total    = np.zeros(N)
    np.add.at(same_cnt, row_idx[same_m], 1)
    np.add.at(total,    row_idx,         1)
    same_frac = same_cnt / np.maximum(total, 1)

    hardness = (off_ratio + (1.0 - diag_dom) + (1.0 - same_frac)) / 3.0
    t_min, t_max = hardness.min(), hardness.max()
    return ((hardness - t_min) / (t_max - t_min + 1e-8)).astype(np.float32)


def convergence_targets(
    A: sp.csr_matrix,
    P: sp.csr_matrix,
    membership: np.ndarray,
    num_trials: int = 8,
    smoother_steps: int = 6,
    smoother_weight: float = 0.8,
) -> np.ndarray:
    """
    Per-node V-cycle error reduction factor — theoretically correct target.
    Falls back to proxy_targets if LU factorization fails.
    """
    from scipy.sparse.linalg import splu

    N  = A.shape[0]
    Ac = (P.T @ A @ P).tocsc()
    try:
        Ac_lu = splu(Ac)
    except Exception as e:
        print(f"    LU failed ({e}), using proxy targets.")
        return proxy_targets(A, P, membership)

    D_inv = 1.0 / (A.diagonal() + 1e-12)
    rng   = np.random.default_rng(42)
    node_factors = np.zeros((num_trials, N))

    for t in range(num_trials):
        e        = rng.standard_normal(N)
        e_before = np.abs(e).copy()

        # Pre-smooth (zero RHS — pure error propagation)
        for _ in range(smoother_steps):
            e = e + smoother_weight * D_inv * (-A @ e)
        # Coarse correction
        r_c  = P.T @ (-A @ e)
        e   += P @ Ac_lu.solve(r_c)
        # Post-smooth
        for _ in range(smoother_steps):
            e = e + smoother_weight * D_inv * (-A @ e)

        node_factors[t] = np.abs(e) / (e_before + 1e-12)

    mean_factors = node_factors.mean(axis=0)
    t_min, t_max = mean_factors.min(), mean_factors.max()
    print(
        f"    conv targets: global={mean_factors.mean():.4f}  "
        f"min={mean_factors.min():.4f}  max={mean_factors.max():.4f}"
    )
    return ((mean_factors - t_min) / (t_max - t_min + 1e-8)).astype(np.float32)


# ============================================================
# SAMPLE BUILDERS
# ============================================================

def build_mlp_samples(A, membership, num_clusters, target_mode):
    """Flat (features, target) pairs for MLP training."""
    feats   = extract_node_features(A, membership, num_clusters)
    P       = build_classical_P(A, membership, num_clusters)
    targets = (
        convergence_targets(A, P, membership)
        if target_mode == "convergence"
        else proxy_targets(A, P, membership)
    )
    N = A.shape[0]
    return [(feats[i], float(targets[i])) for i in range(N)]


def build_gnn_graph(A, membership, num_clusters, target_mode):
    """Graph dict for GNN training (node features + edge structure + targets)."""
    N    = A.shape[0]
    diag = np.abs(A.diagonal())

    # Node features
    node_features = extract_node_features(A, membership, num_clusters)

    # Edge index and attributes (vectorized, no diagonal)
    A_coo    = A.tocoo()
    row_idx  = A_coo.row
    col_idx  = A_coo.col
    vals     = A_coo.data
    off_mask = row_idx != col_idx

    src        = row_idx[off_mask]
    dst        = col_idx[off_mask]
    abs_vals   = np.abs(vals[off_mask])
    norm_vals  = abs_vals / (diag[src] + 1e-12)
    same_flags = (membership[src] == membership[dst]).astype(np.float32)

    edge_index = np.array([src, dst], dtype=np.int64)
    edge_attr  = np.column_stack([abs_vals, norm_vals, same_flags]).astype(np.float32)

    # Targets
    P = build_classical_P(A, membership, num_clusters)
    targets = (
        convergence_targets(A, P, membership)
        if target_mode == "convergence"
        else proxy_targets(A, P, membership)
    )

    return {
        "node_features": node_features,
        "edge_index":    edge_index,
        "edge_attr":     edge_attr,
        "targets":       targets,
        "membership":    membership,
        "num_clusters":  num_clusters,
        "N":             N,
    }


# ============================================================
# LOAD / VALIDATE / REPAIR (same logic as mlP_dataset_generator_suitesparse.py)
# ============================================================

def load_matrix(path):
    try:
        if path.endswith(".mtx"):
            A = scipy.io.mmread(path)
        elif path.endswith(".mat"):
            mat = scipy.io.loadmat(path)
            if "Problem" in mat:
                A = mat["Problem"][0, 0]["A"]
            else:
                for val in mat.values():
                    if sp.issparse(val):
                        A = val
                        break
                else:
                    return None
        else:
            return None
        return sp.csr_matrix(A, dtype=np.float64)
    except Exception as e:
        print(f"  Load error: {e}")
        return None


def is_valid_spd(A, min_size, max_size):
    if A.shape[0] != A.shape[1]:
        return False, "Not square"
    n = A.shape[0]
    if n < min_size:
        return False, f"Too small ({n})"
    if n > max_size:
        return False, f"Too large ({n})"
    if np.any(A.diagonal() <= 0):
        return False, "Non-positive diagonal"
    rng  = np.random.default_rng(42)
    rows = rng.integers(0, n, 200)
    cols = rng.integers(0, n, 200)
    mask = rows != cols
    rows, cols = rows[mask], cols[mask]
    A_csr   = A.tocsr()
    vij = np.array(A_csr[rows, cols]).ravel()
    vji = np.array(A_csr[cols, rows]).ravel()
    err = np.max(np.abs(vij - vji) / (np.abs(vij) + 1e-12))
    if err > 0.1:
        return False, f"Not symmetric (err={err:.3f})"
    return True, "OK"


def make_spd(A):
    A = (A + A.T) / 2.0
    d = A.diagonal().min()
    if d <= 0:
        A = A + (abs(d) + 1e-6) * sp.eye(A.shape[0])
    return A.tocsr()


def run_metis(A, num_clusters):
    adj = (A + A.T).tocsr()
    adj.setdiag(0)
    adj.eliminate_zeros()
    effective = min(num_clusters, A.shape[0] // 4)
    if effective < 4:
        return None
    _, mem = pymetis.part_graph(
        effective,
        adjacency=pymetis.CSRAdjacency(adj.indptr, adj.indices),
    )
    return np.array(mem), effective


# ============================================================
# MAIN GENERATOR
# ============================================================

def generate(
    matrix_dir: str,
    output: str,
    mode: str = "gnn",
    target_mode: str = "convergence",
    num_clusters: int = 64,
    min_size: int = 500,
    max_size: int = 50_000,
    checkpoint_every: int = 10,
):
    mtx_files = []
    for root, _, files in os.walk(matrix_dir):
        for fname in sorted(files):
            if fname.endswith(".mtx") and not fname.endswith("_b.mtx"):
                mtx_files.append(os.path.join(root, fname))

    print("=" * 60)
    print(f"Dataset Generator  mode={mode}  target={target_mode}")
    print("=" * 60)
    print(f"  .mtx files : {len(mtx_files)}")
    print(f"  Size filter: [{min_size}, {max_size}]")
    print(f"  Output     : {output}")
    print("=" * 60)

    all_data  = []
    n_ok = n_fail = 0

    for i, path in enumerate(mtx_files):
        name = os.path.basename(path)
        print(f"\n[{i+1}/{len(mtx_files)}] {name}")

        A = load_matrix(path)
        if A is None:
            n_fail += 1
            continue

        valid, reason = is_valid_spd(A, min_size, max_size)
        if not valid:
            print(f"  ✗ {reason}")
            continue

        A = make_spd(A)
        print(f"  ✓ shape={A.shape}, nnz={A.nnz}")

        result = run_metis(A, num_clusters)
        if result is None:
            print("  ✗ Too few nodes for clustering")
            n_fail += 1
            continue
        membership, effective = result

        t0 = time.time()
        try:
            if mode == "mlp":
                data = build_mlp_samples(A, membership, effective, target_mode)
            else:
                data = build_gnn_graph(A, membership, effective, target_mode)
        except Exception as e:
            print(f"  ✗ Processing error: {e}")
            n_fail += 1
            continue

        if mode == "mlp":
            all_data.extend(data)
            n_samples = len(data)
        else:
            all_data.append(data)
            n_samples = data["N"]

        n_ok += 1
        print(f"  ✓ samples={n_samples}  t={time.time()-t0:.1f}s  total_items={len(all_data)}")

        if n_ok % checkpoint_every == 0:
            ckpt = output.replace(".pkl", f"_ckpt_{n_ok}.pkl")
            with open(ckpt, "wb") as f:
                pickle.dump(all_data, f)
            print(f"  → Checkpoint: {ckpt}")

    print(f"\n{'='*60}")
    print(f"Done — {n_ok} matrices processed, {n_fail} failed/skipped")

    if all_data:
        with open(output, "wb") as f:
            pickle.dump(all_data, f)
        print(f"✓ Dataset saved: {output}")

        if mode == "mlp":
            X = np.array([d[0] for d in all_data], dtype=np.float32)
            y = np.array([d[1] for d in all_data], dtype=np.float32)
            print(f"  Samples  : {X.shape[0]}")
            print(f"  Features : {X.shape[1]}")
            print(f"  Targets  : min={y.min():.4f}  max={y.max():.4f}  mean={y.mean():.4f}  std={y.std():.4f}")
        else:
            total_nodes = sum(g["N"] for g in all_data)
            total_edges = sum(g["edge_index"].shape[1] for g in all_data)
            all_t = np.concatenate([g["targets"] for g in all_data])
            print(f"  Graphs   : {len(all_data)}")
            print(f"  Nodes    : {total_nodes}")
            print(f"  Edges    : {total_edges}")
            print(f"  Targets  : min={all_t.min():.4f}  max={all_t.max():.4f}  mean={all_t.mean():.4f}  std={all_t.std():.4f}")
    else:
        print("✗ No data collected.")

    return all_data


# ============================================================
# MERGE UTILITY (matches mlP_dataset_generator_suitesparse.py)
# ============================================================

def merge_datasets(inputs, output):
    all_data = []
    for path in inputs:
        if not os.path.exists(path):
            print(f"  Warning: {path} not found, skipping.")
            continue
        with open(path, "rb") as f:
            d = pickle.load(f)
        all_data.extend(d)
        print(f"  Loaded {len(d)} items from {path}")
    with open(output, "wb") as f:
        pickle.dump(all_data, f)
    print(f"✓ Merged {len(all_data)} items → {output}")
    return all_data


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # --------------------------------------------------------
    # *** CONFIGURE HERE — change these as needed ***
    # --------------------------------------------------------
    MATRIX_DIR       = "/home/xallka/Escritorio/BSC/PythonCode/CheckAMG_precond2/mlP_suitesparse/suitesparse_matrices"
    MODE             = "gnn"    # "gnn" or "mlp"
    TARGET           = "proxy"  # "proxy" or "convergence"
    NUM_CLUSTERS     = 64
    MIN_SIZE         = 500
    MAX_SIZE         = 50_000
    CHECKPOINT_EVERY = 10
    # --------------------------------------------------------

    THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.join(THIS_DIR, "datasets")
    os.makedirs(DATASET_DIR, exist_ok=True)

    if MODE == "gnn" and TARGET == "convergence":
        output = os.path.join(DATASET_DIR, "gnn_dataset_conv.pkl")
    elif MODE == "gnn" and TARGET == "proxy":
        output = os.path.join(DATASET_DIR, "gnn_dataset_proxy.pkl")
    else:
        output = os.path.join(DATASET_DIR, "mlp_dataset.pkl")

    print(f"Matrix dir : {MATRIX_DIR}")
    print(f"Output     : {output}")

    generate(
        matrix_dir=MATRIX_DIR,
        output=output,
        mode=MODE,
        target_mode=TARGET,
        num_clusters=NUM_CLUSTERS,
        min_size=MIN_SIZE,
        max_size=MAX_SIZE,
        checkpoint_every=CHECKPOINT_EVERY,
    )