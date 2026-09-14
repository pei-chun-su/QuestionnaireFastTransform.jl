"""flip_questionnaire.py -- canonical implementation of the Questionnaire with the FLIP PERTURBATION
(Algorithm 3, main_revised0803.tex, section "Flip Perturbation").

THE FLIP PERTURBATION (the correct algorithm -- do NOT use the old aff_from_order/1-D-Gaussian shortcut):
  every tau iterations, build the tree T from the current dual affinity as usual, then PERMUTE THE POINTS
  AT A UNIFORMLY RANDOM HALF OF THE LEAVES OF T (holding the rest in place), and induce the dual
  *tree* affinity from that PERTURBED tree. We never reconstruct a 1-D Gaussian affinity from a flipped
  leaf order; we flip the leaves of the actual tree and use its tree affinity.

Key identity used here: relabeling a fraction of a tree's leaves == permuting that fraction of the data's
point indices and re-using the SAME tree structure. So the perturbed dual affinity is simply
    dual_affinity_from_tree(mat[b], tree)
where b permutes a uniformly random `frac` of the point indices (identity on the rest). Proof: a leaf that
held point p now holds point q; the folder that contained p now contains q; averaging the data over the
perturbed folders equals averaging mat[b] (row p := mat[b[p]]) over the ORIGINAL folders.
"""
import os, sys
import numpy as np

_PYQ = os.path.dirname(os.path.abspath(__file__))
if _PYQ not in sys.path:
    sys.path.insert(0, _PYQ)
sys.modules.setdefault('cupy', np)                       # pyquest imports cupy; stub with numpy on CPU
import bin_tree_build, dual_affinity, questionnaire as _Q
from sklearn.metrics.pairwise import cosine_similarity


def build_tree(aff, seed=0):
    """Binary partition tree from an affinity (r_dyadic cut is seeded for determinism)."""
    np.random.seed(seed)
    root, _, _ = bin_tree_build.bin_tree_build4(aff)
    return root


def dual_affinity_from_tree(mat, tree, alpha=1.0, signed=True):
    """W_T: the dual (tree) affinity that `tree` (on the rows of `mat`) induces on the columns of `mat`.
    removemean=0 is the cosine/correlation dual (DUAL_COSINE); signed=True clips negatives (max(cos,0))."""
    W, _, _ = dual_affinity.partition_dualgeometry(np.expand_dims(mat, 2), tree, alpha, 0, signed)
    return W


def flip_tree_leaves_dual(mat, tree, rng, frac=0.5, alpha=1.0, signed=True):
    """FLIP PERTURBATION: permute the points at a uniformly random `frac` of the leaves of `tree`
    (holding the rest in place) and return the dual affinity the PERTURBED tree induces on the columns
    of `mat`.  Equivalent to permuting that fraction of the point indices of `mat` and reusing `tree`."""
    n = mat.shape[0]
    b = np.arange(n)
    k = int(round(frac * n))
    m = rng.choice(n, k, replace=False)
    b[m] = b[rng.permutation(m)]
    return dual_affinity_from_tree(mat[b], tree, alpha, signed)


def curve_order(aff, seed=0):
    """Space-filling-curve (flip-free optimal leaf) order for an affinity."""
    np.random.seed(seed)
    root, _, _ = bin_tree_build.bin_tree_build4(aff)
    return np.array(_Q._sdq_reorient_order(root, aff))


def flip_questionnaire(K, W_X0=None, T=200, tau=20, frac=0.5, alpha=1.0, signed=True, seed=0,
                       verbose=False, record_diffs=False):
    """Questionnaire with flip perturbation (Algorithm 3), column-first (seed W_X^0).

    K      : data matrix, rows = Y, cols = X.
    W_X0   : initial column affinity (seed). Default: clip(cosine(K.T), 0).
    T      : number of iterations.
    tau    : perturbation period; every tau-th iteration flips half the leaves on each axis.
             Use tau=None / 0 / np.inf for the RAW questionnaire (no perturbation).
    frac   : perturbed fraction of leaves (0.5 = half).
    Returns dict(row_aff, col_aff, row_order, col_order).
    """
    prng = np.random.default_rng(seed)
    col_aff = np.clip(cosine_similarity(K.T), 0, None) if W_X0 is None else np.asarray(W_X0, float)
    row_aff = np.clip(cosine_similarity(K), 0, None)
    pert = (tau is not None) and (tau != 0) and np.isfinite(float(tau))
    res_row, res_col, pert_iters = [], [], []
    for t in range(T):
        do = pert and (t > 0) and (t % int(tau) == 0)
        if do:
            pert_iters.append(t + 1)
        col_tree = build_tree(col_aff)                                                  # T_X from W_X^{t-1}
        new_row = (flip_tree_leaves_dual(K.T, col_tree, prng, frac, alpha, signed) if do
                   else dual_affinity_from_tree(K.T, col_tree, alpha, signed))          # (flip T_X) -> W_Y = W_{T_X}
        if record_diffs:
            res_row.append(float(np.abs(new_row - row_aff).max()))                      # ||W_Y^t - W_Y^{t-1}||_inf
        row_aff = new_row
        row_tree = build_tree(row_aff)                                                  # T_Y from W_Y
        new_col = (flip_tree_leaves_dual(K, row_tree, prng, frac, alpha, signed) if do
                   else dual_affinity_from_tree(K, row_tree, alpha, signed))            # (flip T_Y) -> W_X = W_{T_Y}
        if record_diffs:
            res_col.append(float(np.abs(new_col - col_aff).max()))                      # ||W_X^t - W_X^{t-1}||_inf
        col_aff = new_col
        if verbose and (t + 1) % 20 == 0:
            print("iter %d/%d%s" % (t + 1, T, "  [flip]" if do else ""), flush=True)
    out = dict(row_aff=row_aff, col_aff=col_aff,
               row_order=curve_order(row_aff), col_order=curve_order(col_aff))
    if record_diffs:
        out.update(res_row=np.array(res_row), res_col=np.array(res_col), pert_iters=np.array(pert_iters))
    return out
