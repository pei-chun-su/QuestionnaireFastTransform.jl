"""No questionnaire: build the sphere tree DIRECTLY on X from a Euclidean-distance affinity (self-tuning
kNN Gaussian), use it as both row and col tree, then one tree-EMD for the (l,m) channel geometry.
N=4096, 99 harmonics (l=1..9), data-driven east pole. Colored by l-|m|, |m|, and SFC order."""
import os, sys, time, numpy as np
from math import factorial, pi, sqrt
import scipy.special as sp
from scipy.spatial.distance import cdist
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pyquest"))
import flip_questionnaire as FQ
import bin_tree_build, dual_affinity
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import BoundaryNorm, ListedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa
matplotlib.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'text.latex.preamble': r'\usepackage{amsmath,amssymb}'})
HERE = os.path.dirname(os.path.abspath(__file__)); N = 4096; Lmax = 9; KNN = 10
t0 = time.time()

rng = np.random.default_rng(10); X = rng.standard_normal((N, 3)); X /= np.linalg.norm(X, axis=1, keepdims=True)
# ---- tree DIRECTLY on X: self-tuning Gaussian on Euclidean distance ----
Deuc = cdist(X, X)                                                     # 3D Euclidean distances
sig = np.sort(Deuc, axis=1)[:, KNN]                                    # local scale = dist to KNN-th neighbor
W = np.exp(-(Deuc ** 2) / (sig[:, None] * sig[None, :])); np.fill_diagonal(W, 0.0)
tree = bin_tree_build.bin_tree_build(W, 'r_dyadic', 1.0, False)        # the X-tree (row = col)
print("[euc] X-tree built directly from Euclidean affinity (size=%d, depth=%d, %.0fs)" % (tree.size, tree.tree_depth, time.time()-t0), flush=True)

cosang = np.clip(X @ X.T, -1.0, 1.0)                                   # data-driven east pole
absc = np.abs(cosang).copy(); np.fill_diagonal(absc, np.inf); jstar = np.argmin(absc, axis=1)
east = X[jstar]; e1 = east - (np.sum(east * X, 1)[:, None]) * X; n1 = np.linalg.norm(e1, axis=1)
bad = n1 < 1e-9
if bad.any():
    alt = np.cross(X[bad], [0., 0., 1.]); e1[bad] = alt; n1[bad] = np.linalg.norm(alt, axis=1)
e1 /= n1[:, None]; e2 = np.cross(X, e1); phi = np.arctan2(e2 @ X.T, e1 @ X.T)
chans = [(l, m) for l in range(1, Lmax + 1) for m in range(-l, l + 1)]
larr = np.array([l for l, m in chans]); marr = np.array([m for l, m in chans]); aml = np.abs(marr); nlat = larr - aml; NC = len(chans)
print("[euc] %d harmonics; building tensor %dx%dx%d (float32) + one tree-EMD ..." % (NC, N, N, NC), flush=True)
T = np.empty((N, N, NC), dtype=np.float32)                            # ~6.6 GB
for k, (l, m) in enumerate(chans):
    am = abs(m); Nlm = sqrt((2*l+1)/(4*pi)*factorial(l-am)/factorial(l+am)); Plm = sp.lpmv(am, l, cosang)
    if m > 0:   T[:, :, k] = (sqrt(2)*Nlm*Plm*np.cos(am*phi)).astype(np.float32)
    elif m < 0: T[:, :, k] = (sqrt(2)*Nlm*Plm*np.sin(am*phi)).astype(np.float32)
    else:       T[:, :, k] = (Nlm*Plm).astype(np.float32)
print("[euc] tensor built (%.0fs); calc_2demd ..." % (time.time()-t0), flush=True)
chan_emd = dual_affinity.calc_2demd(T, tree, tree, row_alpha=0.0, row_beta=1.0, col_alpha=0.0, col_beta=1.0)
chan_aff = np.asarray(dual_affinity.emd_dual_aff(chan_emd)); del T
print("[euc] channel EMD done (%.0fs)" % (time.time()-t0), flush=True)

def cpos(o):
    p = np.empty(len(o), int); p[o] = np.arange(len(o)); return p
def lap_embed(Wm, k=3):
    Wm = np.array(Wm, float); np.fill_diagonal(Wm, 0.0); dd = Wm.sum(1)+1e-12; DM = 1.0/np.sqrt(dd)
    Ws = DM[:, None]*Wm*DM[None, :]; v, V = np.linalg.eigh(Ws); return V[:, list(range(-2, -2-k, -1))]
Ech = lap_embed(chan_aff, 3); chsfc = cpos(np.asarray(FQ.curve_order(chan_aff)))
mc = lambda lab: max(abs(np.corrcoef(Ech[:, d], lab)[0, 1]) for d in range(3))
rL, rAM, rNL, rM = mc(larr), mc(aml), mc(nlat), mc(marr)
print("[euc] channel |corr|: l=%.3f ; |m|=%.3f ; l-|m|=%.3f ; sign m=%.3f" % (rL, rAM, rNL, rM), flush=True)
so = np.asarray(FQ.curve_order(W))                                    # sphere SFC from the Euclidean tree affinity
np.savez_compressed(os.path.join(HERE, "quest_euclid_tree_data.npz"), X=X.astype(np.float32), chan_aff=chan_aff,
                    larr=larr, marr=marr, Ech=Ech, sphere_order=so)

pair = ['(%d,%d)' % (l, m) for l, m in chans]
big = ListedColormap((list(mpl.colormaps['tab10'].colors) + list(mpl.colormaps['Set1'].colors))[:Lmax+1]); n08 = BoundaryNorm(np.arange(-0.5, Lmax+1.5, 1.0), Lmax+1)
fig = plt.figure(figsize=(21, 7.2))
def pan(pos, c, cm, nm, tk, cblab, title):
    ax = fig.add_subplot(1, 3, pos, projection='3d')
    kw = dict(s=32, cmap=cm, alpha=.97, depthshade=False, edgecolors='k', linewidths=.3)
    if nm is not None: kw['norm'] = nm
    sc = ax.scatter(Ech[:, 0], Ech[:, 1], Ech[:, 2], c=c, **kw)
    for i in range(NC): ax.text(Ech[i, 0], Ech[i, 1], Ech[i, 2], pair[i], fontsize=4.2, ha='center', va='center')
    ax.view_init(elev=25, azim=90); ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([]); ax.set_title(title, fontsize=13)
    fig.colorbar(sc, ax=ax, shrink=.6, pad=.02, **({'ticks': list(tk)} if tk is not None else {})).set_label(cblab, fontsize=11)
pan(1, nlat, big, n08, range(0, Lmax+1), r'$\ell-|m|$ (parallels)', r'$\ell-|m|$ ($|r|=%.2f$)' % rNL)
pan(2, aml, big, n08, range(0, Lmax+1), r'$|m|$ (meridians)', r'$|m|$ ($|r|=%.2f$)' % rAM)
pan(3, chsfc, 'turbo', None, None, 'channel curve order', r'space-filling curve order')
fig.suptitle(r'Direct Euclidean tree on $X$ (no questionnaire), $N=%d$, 99 harmonics: $(\ell,m)$ channel embedding --- $|m|\ %.2f$, $\ell{-}|m|\ %.2f$, $\ell\ %.2f$' % (N, rAM, rNL, rL), fontsize=13, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(HERE, 'quest_euclid_tree.png'), dpi=140, bbox_inches='tight')
fig.savefig(os.path.join(HERE, 'quest_euclid_tree.pdf'), bbox_inches='tight'); print("SAVED; done (%.0fs)" % (time.time()-t0), flush=True)
