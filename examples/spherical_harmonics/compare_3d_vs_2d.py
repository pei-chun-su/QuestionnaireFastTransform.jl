"""Compare two ways to get the (l,m) channel geometry of the SH tensor T[i,j,(l,m)] = Y_l^m(theta_ij,phi_ij),
X=Y, data-driven east pole (nearest-orthogonal point).
  Version A  (full 3D):        pyquest3d(T)  -> channel affinity from the iterated 3D questionnaire.
  Version B  (2D-then-EMD):    2D questionnaire on the inner-product Gram K=X.X^T  -> row/col trees;
                               then ONE calc_2demd(T, row_tree, col_tree) for the (l,m) channel affinity.
Both use the SAME X and the SAME tensor. Reports |corr| of the channel embedding with l, |m|, l-|m|, sign m."""
import os, sys, time, numpy as np
from math import factorial, pi, sqrt
import scipy.special as sp
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pyquest"))
import flip_questionnaire as FQ
import questionnaire as Q
import dual_affinity
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import BoundaryNorm, ListedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa
matplotlib.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'text.latex.preamble': r'\usepackage{amsmath,amssymb}'})
HERE = os.path.dirname(os.path.abspath(__file__)); N = 512; Lmax = 7; NIT = 10
t0 = time.time()

# ---- sphere, data-driven east pole, SH tensor (shared by both versions) ----
rng = np.random.default_rng(10); X = rng.standard_normal((N, 3)); X /= np.linalg.norm(X, axis=1, keepdims=True)
cosang = np.clip(X @ X.T, -1.0, 1.0)
absc = np.abs(cosang).copy(); np.fill_diagonal(absc, np.inf); jstar = np.argmin(absc, axis=1)
east = X[jstar]; e1 = east - (np.sum(east * X, 1)[:, None]) * X; n1 = np.linalg.norm(e1, axis=1)
bad = n1 < 1e-9
if bad.any():
    alt = np.cross(X[bad], [0., 0., 1.]); e1[bad] = alt; n1[bad] = np.linalg.norm(alt, axis=1)
e1 /= n1[:, None]; e2 = np.cross(X, e1); phi = np.arctan2(e2 @ X.T, e1 @ X.T)
chans = [(l, m) for l in range(1, Lmax + 1) for m in range(-l, l + 1)]
larr = np.array([l for l, m in chans]); marr = np.array([m for l, m in chans]); aml = np.abs(marr); nlat = larr - aml; NC = len(chans)
T = np.empty((N, N, NC))
for k, (l, m) in enumerate(chans):
    am = abs(m); Nlm = sqrt((2*l+1)/(4*pi)*factorial(l-am)/factorial(l+am)); Plm = sp.lpmv(am, l, cosang)
    if m > 0:   T[:, :, k] = sqrt(2)*Nlm*Plm*np.cos(am*phi)
    elif m < 0: T[:, :, k] = sqrt(2)*Nlm*Plm*np.sin(am*phi)
    else:       T[:, :, k] = Nlm*Plm
Tp = T - T.min() + 0.01
print("[cmp] X, tensor built (%s), east-pole mean|cos|=%.4f" % (T.shape, np.abs(cosang[np.arange(N), jstar]).mean()), flush=True)

def lap_embed(Wm, k=3):
    Wm = np.array(Wm, float); np.fill_diagonal(Wm, 0.0); dd = Wm.sum(1)+1e-12; DM = 1.0/np.sqrt(dd)
    Ws = DM[:, None]*Wm*DM[None, :]; v, V = np.linalg.eigh(Ws); return V[:, list(range(-2, -2-k, -1))]
def corrs(aff):
    E = lap_embed(aff, 3)
    mc = lambda lab: max(abs(np.corrcoef(E[:, d], lab)[0, 1]) for d in range(3))
    return E, {'l': mc(larr), '|m|': mc(aml), 'l-|m|': mc(nlat), 'sign m': mc(marr)}

# ---- Version A: full 3D questionnaire ----
print("[cmp] Version A: full pyquest3d ...", flush=True)
pA = Q.PyQuest3DParams(Q.INIT_AFF_COS_SIM, Q.TREE_TYPE_BINARY, Q.DUAL_EMD, Q.DUAL_EMD, Q.DUAL_EMD, n_iters=NIT)
pA.chan_tree_constant = 1.0
runA = Q.pyquest3d(Tp, pA); chanA = np.asarray(runA.chan_aff)
EA, cA = corrs(chanA); print("[cmp] A (full 3D)   %s (%.0fs)" % (cA, time.time()-t0), flush=True)

# ---- Version B: 2D questionnaire on X.X^T, then one calc_2demd through those trees ----
print("[cmp] Version B: 2D questionnaire on Gram + calc_2demd ...", flush=True)
K = (X @ X.T).astype(float) + 1.0
pB = Q.PyQuestParams(Q.INIT_AFF_COS_SIM, Q.TREE_TYPE_BINARY, Q.DUAL_EMD, Q.DUAL_EMD, n_iters=NIT)
runB = Q.pyquest(K, pB); rtree = runB.row_trees[-1]; ctree = runB.col_trees[-1]
chan_emd = dual_affinity.calc_2demd(Tp, rtree, ctree, row_alpha=0.0, row_beta=1.0, col_alpha=0.0, col_beta=1.0)
chanB = np.asarray(dual_affinity.emd_dual_aff(chan_emd))
EB, cB = corrs(chanB); print("[cmp] B (2D+EMD)    %s (%.0fs)" % (cB, time.time()-t0), flush=True)
np.savez_compressed(os.path.join(HERE, "compare_3d_vs_2d_data.npz"), X=X.astype(np.float32),
                    chanA=chanA, chanB=chanB, EA=EA, EB=EB, larr=larr, marr=marr, row_affB=np.asarray(runB.row_aff))

# ---- figure: channel embeddings by l-|m|, side by side ----
pair = ['(%d,%d)' % (l, m) for l, m in chans]
norm08 = BoundaryNorm(np.arange(-0.5, Lmax+1.5, 1.0), Lmax+1)
cpol = ListedColormap(mpl.colormaps['tab10'].colors[:Lmax+1])
fig = plt.figure(figsize=(17, 8.5))
def pan(pos, E, c, title):
    ax = fig.add_subplot(1, 2, pos, projection='3d')
    sc = ax.scatter(E[:, 0], E[:, 1], E[:, 2], c=c, s=52, cmap=cpol, norm=norm08, alpha=.98, depthshade=False, edgecolors='k', linewidths=.4)
    for i in range(NC): ax.text(E[i, 0], E[i, 1], E[i, 2], pair[i], fontsize=6, ha='center', va='center')
    ax.view_init(elev=65, azim=324); ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([]); ax.set_title(title, fontsize=14)
    fig.colorbar(sc, ax=ax, shrink=.6, pad=.02, ticks=range(0, Lmax+1)).set_label(r'$\ell-|m|$ (parallels)', fontsize=12)
pan(1, EA, nlat, r'\textbf{A. Full 3D questionnaire} --- $\ell{-}|m|\ |r|=%.2f$ ($\ell\,{=}%.2f$, $|m|{=}%.2f$)' % (cA['l-|m|'], cA['l'], cA['|m|']))
pan(2, EB, nlat, r'\textbf{B. 2D on }$X\!\cdot\!X$\textbf{ then EMD} --- $\ell{-}|m|\ |r|=%.2f$ ($\ell\,{=}%.2f$, $|m|{=}%.2f$)' % (cB['l-|m|'], cB['l'], cB['|m|']))
fig.suptitle(r'Full 3D questionnaire vs.\ 2D-on-$X\!\cdot\!X$-then-EMD for the $(\ell,m)$ channel geometry (colored by $\ell-|m|$; $N=%d$, $\ell\le%d$)' % (N, Lmax), fontsize=14, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(HERE, 'compare_3d_vs_2d.png'), dpi=140, bbox_inches='tight')
fig.savefig(os.path.join(HERE, 'compare_3d_vs_2d.pdf'), bbox_inches='tight'); print("SAVED figure")
print("done (%.0fs)" % (time.time()-t0), flush=True)
