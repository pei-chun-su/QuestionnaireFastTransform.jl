"""Version B at N=2048: 2D questionnaire on the inner-product Gram K=X.X^T, then ONE tree-EMD (calc_2demd)
for the (l,m) channel geometry of T[i,j,(l,m)]=Y_l^m(theta_ij,phi_ij) with the data-driven east pole.
No 3D questionnaire. Signed tensor used directly (tree-EMD is shift-invariant). Channel embedding shown
colored by l-|m| and by the channel space-filling curve."""
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
HERE = os.path.dirname(os.path.abspath(__file__)); N = 2048; Lmax = 7; NIT = 10
t0 = time.time()

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
T = np.empty((N, N, NC))                                                # signed tensor, ~2.1 GB
for k, (l, m) in enumerate(chans):
    am = abs(m); Nlm = sqrt((2*l+1)/(4*pi)*factorial(l-am)/factorial(l+am)); Plm = sp.lpmv(am, l, cosang)
    if m > 0:   T[:, :, k] = sqrt(2)*Nlm*Plm*np.cos(am*phi)
    elif m < 0: T[:, :, k] = sqrt(2)*Nlm*Plm*np.sin(am*phi)
    else:       T[:, :, k] = Nlm*Plm
print("[B2048] X + tensor %s built (%.0fs), east mean|cos|=%.4f" % (T.shape, time.time()-t0, np.abs(cosang[np.arange(N), jstar]).mean()), flush=True)

K = (X @ X.T).astype(float) + 1.0
pB = Q.PyQuestParams(Q.INIT_AFF_COS_SIM, Q.TREE_TYPE_BINARY, Q.DUAL_EMD, Q.DUAL_EMD, n_iters=NIT)
print("[B2048] running 2D questionnaire on Gram ...", flush=True)
runB = Q.pyquest(K, pB); rtree = runB.row_trees[-1]; ctree = runB.col_trees[-1]; row_aff = np.asarray(runB.row_aff)
print("[B2048] 2D questionnaire done (%.0fs); running calc_2demd for (l,m) channels ..." % (time.time()-t0), flush=True)
chan_emd = dual_affinity.calc_2demd(T, rtree, ctree, row_alpha=0.0, row_beta=1.0, col_alpha=0.0, col_beta=1.0)
chan_aff = np.asarray(dual_affinity.emd_dual_aff(chan_emd))
print("[B2048] channel EMD done (%.0fs)" % (time.time()-t0), flush=True)

def cpos(o):
    p = np.empty(len(o), int); p[o] = np.arange(len(o)); return p
def lap_embed(Wm, k=3):
    Wm = np.array(Wm, float); np.fill_diagonal(Wm, 0.0); dd = Wm.sum(1)+1e-12; DM = 1.0/np.sqrt(dd)
    Ws = DM[:, None]*Wm*DM[None, :]; v, V = np.linalg.eigh(Ws); return V[:, list(range(-2, -2-k, -1))]
Ech = lap_embed(chan_aff, 3); chsfc = cpos(np.asarray(FQ.curve_order(chan_aff)))
mc = lambda lab: max(abs(np.corrcoef(Ech[:, d], lab)[0, 1]) for d in range(3))
rL, rAM, rNL, rM = mc(larr), mc(aml), mc(nlat), mc(marr)
print("[B2048] channel |corr|: l=%.3f ; |m|=%.3f ; l-|m|=%.3f ; sign m=%.3f" % (rL, rAM, rNL, rM), flush=True)
np.savez_compressed(os.path.join(HERE, "quest_B_N2048_data.npz"), X=X.astype(np.float32), row_aff=row_aff,
                    chan_aff=chan_aff, larr=larr, marr=marr, Ech=Ech, jstar=jstar)

pair = ['(%d,%d)' % (l, m) for l, m in chans]
cpol = ListedColormap(mpl.colormaps['tab10'].colors[:Lmax+1]); norm08 = BoundaryNorm(np.arange(-0.5, Lmax+1.5, 1.0), Lmax+1)
fig = plt.figure(figsize=(18, 8.4))
def pan(pos, c, cm, nm, tk, cblab, title):
    ax = fig.add_subplot(1, 2, pos, projection='3d')
    sc = ax.scatter(Ech[:, 0], Ech[:, 1], Ech[:, 2], c=c, s=52, cmap=cm, norm=nm, alpha=.98, depthshade=False, edgecolors='k', linewidths=.4)
    for i in range(NC): ax.text(Ech[i, 0], Ech[i, 1], Ech[i, 2], pair[i], fontsize=6, ha='center', va='center')
    ax.view_init(elev=65, azim=324); ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([]); ax.set_title(title, fontsize=13)
    cb = fig.colorbar(sc, ax=ax, shrink=.6, pad=.02, **({'ticks': list(tk)} if tk is not None else {})); cb.set_label(cblab, fontsize=12)
pan(1, nlat, cpol, norm08, range(0, Lmax+1), r'$\ell-|m|$ (parallels)', r'colored by $\ell-|m|$ ($|r|=%.2f$)' % rNL)
pan(2, chsfc, 'turbo', None, None, 'channel curve order', r'colored by space-filling curve order')
fig.suptitle(r'Version B ($2$D on $X\!\cdot\!X$ then EMD), $(\ell,m)$ channel embedding at $N=%d$: $\ell{-}|m|\ |r|=%.2f$, $|m|\ %.2f$, $\ell\ %.2f$' % (N, rNL, rAM, rL), fontsize=14, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(HERE, 'quest_B_N2048.png'), dpi=140, bbox_inches='tight')
fig.savefig(os.path.join(HERE, 'quest_B_N2048.pdf'), bbox_inches='tight'); print("SAVED figure; done (%.0fs)" % (time.time()-t0), flush=True)
