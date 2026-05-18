"""
Helmholtz Kernel Compression Example
=====================================

Demonstrates the full PyQuest pipeline on a Helmholtz Green's function kernel:

1. Generate kernel matrix from two 3D point clouds (thin plane + spiral)
2. Randomly permute rows and columns to destroy spatial ordering
3. Run the Questionnaire algorithm to discover the hidden geometry
4. Reorganize the matrix using the learned tree orderings
5. Compress using Walsh (EGHWT 2D best basis) or Butterfly factorization
6. Evaluate compression quality via matrix-vector multiplication

The key insight is that the Questionnaire algorithm recovers the geometric
structure of the kernel purely from its entries, enabling efficient
hierarchical compression even when the row/column ordering is unknown.
"""

import sys
import os
import time
import pickle
import numpy as np
from scipy.spatial.distance import cdist

# Add the pyquest library to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pyquest'))

import questionnaire as qcoif
from bin_tree_build import bin_tree_build4
from Butterfly import dyadic_partition, compress_dyadic_blocks, apply_compressed_operator
from Walsh import (
    ghwt_core, ghwt_2d_sparse, ghwt_tf_synthesis_2d,
    walsh_multiplication_fast,
)


# ---------------------------------------------------------------------------
# 1. Kernel Generation
# ---------------------------------------------------------------------------

def generate_helmholtz_kernel(n, seed=42):
    """
    Build an n x n Helmholtz kernel matrix K(x,y) = cos(2*pi*|x-y|) / |x-y|
    between a thin plane (source) and a spiral (target) in 3D.

    Parameters
    ----------
    n : int
        Number of points per point cloud.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    matrix : ndarray (n, n)
        The kernel matrix in natural (geometric) ordering.
    z1 : ndarray (3, n)
        Source point cloud (thin plane).
    z2 : ndarray (3, n)
        Target point cloud (spiral).
    """
    rng = np.random.default_rng(seed)

    # Thin plane in 3D
    x_plane = 2.0 * rng.random(n)
    y_plane = 1.0 + 2.0 * rng.random(n)
    z_plane = 0.01 * rng.random(n)
    z1 = np.vstack([x_plane, y_plane, z_plane])

    # Spiral in 3D
    theta = np.linspace(0, 6 * np.pi, n)
    z_spiral = np.linspace(-15, 15, n)
    x_spiral = np.cos(theta)
    y_spiral = np.sin(theta)
    z2 = np.vstack([x_spiral, y_spiral, z_spiral])

    # Pairwise distance and Helmholtz kernel
    D = cdist(z1.T, z2.T, metric='euclidean')
    matrix = np.cos(2 * np.pi * D) / D

    return matrix, z1, z2


# ---------------------------------------------------------------------------
# 2. Questionnaire (Geometry Discovery + Reorganization)
# ---------------------------------------------------------------------------

def run_questionnaire(data, n_iters=5, use_gpu=False):
    """
    Run the PyQuest dual-geometry algorithm to discover row and column trees.

    By default this uses the CPU version (questionnaire.py). Pass use_gpu=True
    (or --gpu on the command line) to use questionnaire_gpu.py instead, which
    accelerates affinity and EMD computation via CuPy/PyTorch on one or more
    GPUs.  The GPU version requires cupy and torch to be installed.

    Parameters
    ----------
    data : ndarray (n, n)
        The (permuted) kernel matrix.
    n_iters : int
        Number of questionnaire iterations.
    use_gpu : bool
        If True, use questionnaire_gpu.py (GPU-accelerated) instead of
        questionnaire.py (CPU-only).

    Returns
    -------
    qrun : PyQuestRun
        The questionnaire result containing row/column trees and affinities.
    quest_time : float
        Wall-clock time in seconds.
    """
    if use_gpu:
        import questionnaire_gpu as qcoif_gpu
        qmod = qcoif_gpu
    else:
        qmod = qcoif

    n = data.shape[0]
    kwargs = {
        "threshold": 0.0,
        "row_alpha": 0.5,
        "col_alpha": 0.5,
        "row_beta": 1.0,
        "col_beta": 1.0,
        "tree_constant": 1.0,
        "knn": int(np.ceil(n ** 0.5)),
        "epsilon": 10,
        "n_iters": n_iters,
        "diag_bias": True,
    }
    if use_gpu:
        kwargs["ngpu"] = 1
        kwargs["ntile"] = 1000

    params = qmod.PyQuestParams(
        qmod.INIT_AFF_COS_SIM,
        qmod.TREE_TYPE_BINARY,
        qmod.DUAL_EMD,
        qmod.DUAL_EMD,
        **kwargs,
    )

    t0 = time.perf_counter()
    qrun = qmod.pyquest(data, params)
    quest_time = time.perf_counter() - t0

    return qrun, quest_time


# ---------------------------------------------------------------------------
# 3. Walsh Compression (EGHWT 2D Best Basis)
# ---------------------------------------------------------------------------

def compress_walsh(data, qrun, acc=1e-3):
    """
    Compress the matrix using the 2D EGHWT (Extended Graph Haar-Walsh Transform)
    best-basis selection.

    Parameters
    ----------
    data : ndarray (n, n)
        The (permuted) kernel matrix.
    qrun : PyQuestRun
        Questionnaire result.
    acc : float
        Energy threshold accuracy (1 - acc^2 of energy is retained).

    Returns
    -------
    dvec_rec : ndarray
        Retained Walsh coefficients.
    infovec_rec : ndarray
        Index pairs for retained coefficients.
    GProws, GPcols : GraphPart
        Graph partition objects for rows and columns.
    walsh_time : float
        Wall-clock time in seconds.
    """
    # Build ordered trees from final affinity
    _, row_order, rs_row = bin_tree_build4(qrun.row_aff)
    _, col_order, rs_col = bin_tree_build4(qrun.col_aff)

    # Compute GHWT core structure
    GProws = ghwt_core(rs_row, row_order)
    GPcols = ghwt_core(rs_col, col_order)

    # 2D sparse best-basis approximation
    energy_threshold = 1.0 - acc ** 2
    t0 = time.perf_counter()
    dvec_rec, infovec_rec, GProws, GPcols = ghwt_2d_sparse(
        data, energy_threshold,
        GProws.rs, GPcols.rs, GProws.ind, GPcols.ind,
    )
    walsh_time = time.perf_counter() - t0

    return dvec_rec, infovec_rec, GProws, GPcols, walsh_time


# ---------------------------------------------------------------------------
# 4. Butterfly Compression (Hierarchical ID Factorization)
# ---------------------------------------------------------------------------

def compress_butterfly(data, qrun, acc=1e-10):
    """
    Compress the matrix using butterfly factorization with interpolative
    decomposition (ID) at each level of the dyadic partition.

    Parameters
    ----------
    data : ndarray (n, n)
        The (permuted) kernel matrix.
    qrun : PyQuestRun
        Questionnaire result.
    acc : float
        ID accuracy tolerance.

    Returns
    -------
    BL : dict
        Top-level skeleton matrices.
    P : dict
        Interpolation matrices at each level.
    level_infos : dict
        Block structure metadata.
    bf_time : float
        Wall-clock time in seconds.
    """
    row_tree = qrun.row_trees[-1]
    col_tree = qrun.col_trees[-1]

    t0 = time.perf_counter()
    level_blocks, level_infos = dyadic_partition(data, row_tree, col_tree, C_max=64)
    BL, P = compress_dyadic_blocks(level_blocks, acc)
    bf_time = time.perf_counter() - t0

    return BL, P, level_infos, bf_time


# ---------------------------------------------------------------------------
# 5. Evaluation: Compressed Matrix-Vector Product
# ---------------------------------------------------------------------------

def evaluate_walsh(data, dvec_rec, infovec_rec, GProws, GPcols, n_trials=100):
    """Evaluate Walsh fast multiplication accuracy and speed."""
    n = data.shape[1]
    rng = np.random.default_rng(0)

    total_time_direct = 0.0
    total_time_walsh = 0.0
    total_err = 0.0

    for _ in range(n_trials):
        x = rng.random(n)

        t0 = time.perf_counter()
        f_direct = data @ x
        total_time_direct += time.perf_counter() - t0

        t0 = time.perf_counter()
        f_walsh = walsh_multiplication_fast(x, dvec_rec, infovec_rec, GProws, GPcols)
        total_time_walsh += time.perf_counter() - t0

        total_err += np.linalg.norm(f_walsh - f_direct) / np.linalg.norm(f_direct)

    return {
        "avg_direct_time": total_time_direct / n_trials,
        "avg_walsh_time": total_time_walsh / n_trials,
        "avg_rel_error": total_err / n_trials,
    }


def evaluate_butterfly(data, BL, P, level_infos, n_trials=100):
    """Evaluate Butterfly fast multiplication accuracy and speed."""
    n = data.shape[1]
    rng = np.random.default_rng(0)

    total_time_direct = 0.0
    total_time_bf = 0.0
    total_err = 0.0

    for _ in range(n_trials):
        x = rng.random(n)

        t0 = time.perf_counter()
        f_direct = data @ x
        total_time_direct += time.perf_counter() - t0

        t0 = time.perf_counter()
        f_bf = apply_compressed_operator(BL, P, x, level_infos)
        total_time_bf += time.perf_counter() - t0

        total_err += np.linalg.norm(f_bf - f_direct) / np.linalg.norm(f_direct)

    return {
        "avg_direct_time": total_time_direct / n_trials,
        "avg_butterfly_time": total_time_bf / n_trials,
        "avg_rel_error": total_err / n_trials,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Helmholtz kernel compression with PyQuest"
    )
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[512, 1024, 2048],
        help="Matrix sizes to test (default: 512 1024 2048)",
    )
    parser.add_argument(
        "--method", choices=["walsh", "butterfly", "both"], default="both",
        help="Compression method (default: both)",
    )
    parser.add_argument(
        "--gpu", action="store_true",
        help="Use GPU-accelerated questionnaire",
    )
    parser.add_argument(
        "--n-iters", type=int, default=5,
        help="Number of questionnaire iterations (default: 5)",
    )
    parser.add_argument(
        "--n-trials", type=int, default=100,
        help="Number of mat-vec trials for evaluation (default: 100)",
    )
    parser.add_argument(
        "--walsh-acc", type=float, default=1e-3,
        help="Walsh energy threshold accuracy (default: 1e-3)",
    )
    parser.add_argument(
        "--butterfly-acc", type=float, default=1e-10,
        help="Butterfly ID accuracy (default: 1e-10)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Helmholtz Kernel Compression with PyQuest")
    print("=" * 70)

    for n in args.sizes:
        print(f"\n{'─' * 70}")
        print(f"  N = {n}  ({n}x{n} matrix)")
        print(f"{'─' * 70}")

        # --- Generate kernel ---
        matrix, z1, z2 = generate_helmholtz_kernel(n)
        print(f"  Kernel generated: ||K|| = {np.linalg.norm(matrix):.4e}")

        # --- Random permutation (simulate unknown ordering) ---
        rng = np.random.default_rng(123)
        row_perm = rng.permutation(n)
        col_perm = rng.permutation(n)
        data = matrix[row_perm, :][:, col_perm]

        # --- Questionnaire ---
        print(f"\n  [1] Running Questionnaire ({args.n_iters} iterations)...")
        qrun, quest_time = run_questionnaire(
            data, n_iters=args.n_iters, use_gpu=args.gpu,
        )
        print(f"      Time: {quest_time:.2f}s")

        # --- Walsh ---
        if args.method in ("walsh", "both"):
            print(f"\n  [2] Walsh compression (acc={args.walsh_acc})...")
            dvec, infovec, GProws, GPcols, walsh_time = compress_walsh(
                data, qrun, acc=args.walsh_acc,
            )
            n_total = n * n
            n_kept = len(dvec)
            ratio = n_total / n_kept
            print(f"      Coefficients: {n_kept:,} / {n_total:,} "
                  f"(ratio: {ratio:.1f}x)")
            print(f"      Compression time: {walsh_time:.2f}s")

            print(f"      Evaluating mat-vec ({args.n_trials} trials)...")
            stats = evaluate_walsh(
                data, dvec, infovec, GProws, GPcols,
                n_trials=args.n_trials,
            )
            print(f"      Direct mat-vec:  {stats['avg_direct_time']:.6f}s")
            print(f"      Walsh mat-vec:   {stats['avg_walsh_time']:.6f}s")
            print(f"      Relative error:  {stats['avg_rel_error']:.2e}")

        # --- Butterfly ---
        if args.method in ("butterfly", "both"):
            print(f"\n  [3] Butterfly compression (acc={args.butterfly_acc})...")
            BL, P, level_infos, bf_time = compress_butterfly(
                data, qrun, acc=args.butterfly_acc,
            )
            # Count total floats stored
            n_floats_BL = sum(v.size for v in BL.values())
            n_floats_P = sum(v.size for v in P.values())
            n_total = n * n
            ratio = n_total / (n_floats_BL + n_floats_P)
            print(f"      Factors: {n_floats_BL + n_floats_P:,} floats "
                  f"(ratio: {ratio:.1f}x)")
            print(f"      Compression time: {bf_time:.2f}s")

            print(f"      Evaluating mat-vec ({args.n_trials} trials)...")
            stats = evaluate_butterfly(
                data, BL, P, level_infos,
                n_trials=args.n_trials,
            )
            print(f"      Direct mat-vec:     {stats['avg_direct_time']:.6f}s")
            print(f"      Butterfly mat-vec:  {stats['avg_butterfly_time']:.6f}s")
            print(f"      Relative error:     {stats['avg_rel_error']:.2e}")

    print(f"\n{'=' * 70}")
    print("Done.")


if __name__ == "__main__":
    main()
