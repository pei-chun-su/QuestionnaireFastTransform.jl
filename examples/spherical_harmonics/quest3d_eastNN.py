"""X=Y 3D questionnaire on T[i,j,(l,m)] = Y_l^m(theta_ij, phi_ij), with a DATA-DRIVEN east pole:
for each x_i the east reference is the sphere point most nearly orthogonal to it,
   j*(i) = argmin_{j!=i} |<x_i,x_j>|,  east_i = x_{j*},
projected into x_i's tangent plane. theta_ij = arccos<x_i,x_j>."""
import os, sys, time, numpy as np
from math import factorial, pi, sqrt
import scipy.special as sp
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
_cc = [0]; _o2 = dual_affinity.calc_2demd
def _tr(*a, **k):
    _cc[0] += 1; print("  calc_2demd #%d (~iter %d/%d, t=%.0fs)" % (_cc[0], (_cc[0]+2)//3, NIT, time.time()-t0), flush=True); return _o2(*a, **k)
dual_affinity.calc_2demd = _tr

rng = np.random.default_rng(10); X = rng.standard_normal((N, 3)); X /= np.linalg.norm(X, axis=1, keepdims=True)
cosang = np.clip(X @ X.T, -1.0, 1.0)                                    # theta_ij
# ---- data-driven east pole: nearest-orthogonal point ----
absc = np.abs(cosang).copy(); np.fill_diagonal(absc, np.inf)           # exclude self
jstar = np.argmin(absc, axis=1)                                        # east reference index per point
east = X[jstar]                                                        # east_i = most-orthogonal sphere point
e1 = east - (np.sum(east * X, 1)[:, None]) * X; n1 = np.linalg.norm(e1, axis=1)
bad = n1 < 1e-9
if bad.any():
    alt = np.cross(X[bad], np.array([0., 0., 1.])); e1[bad] = alt; n1[bad] = np.linalg.norm(alt, axis=1)
e1 /= n1[:, None]; e2 = np.cross(X, e1)                                 # (east, north) tangent frame
phi = np.arctan2(e2 @ X.T, e1 @ X.T)
print("[eastNN] mean |<x_i,x_j*>| = %.4f (0=perfectly orthogonal); building T..." % np.abs(cosang[np.arange(N), jstar]).mean(), flush=True)

chans = [(l, m) for l in range(1, Lmax + 1) for m in range(-l, l + 1)]
larr = np.array([l for l, m in chans]); marr = np.array([m for l, m in chans]); aml = np.abs(marr); nlat = larr - aml; NC = len(chans)
Tt = np.empty((N, N, NC))
for k, (l, m) in enumerate(chans):
    am = abs(m); Nlm = sqrt((2*l+1)/(4*pi)*factorial(l-am)/factorial(l+am)); Plm = sp.lpmv(am, l, cosang)
    if m > 0:   Tt[:, :, k] = sqrt(2)*Nlm*Plm*np.cos(am*phi)
    elif m < 0: Tt[:, :, k] = sqrt(2)*Nlm*Plm*np.sin(am*phi)
    else:       Tt[:, :, k] = Nlm*Plm
Tt = Tt - Tt.min() + 0.01
print("[eastNN] T %s; running pyquest3d NIT=%d..." % (Tt.shape, NIT), flush=True)
params = Q.PyQuest3DParams(Q.INIT_AFF_COS_SIM, Q.TREE_TYPE_BINARY, Q.DUAL_EMD, Q.DUAL_EMD, Q.DUAL_EMD, n_iters=NIT)
params.chan_tree_constant = 1.0
run = Q.pyquest3d(Tt, params)
row_aff = np.asarray(run.row_aff); chan_aff = np.asarray(run.chan_aff)
print("[eastNN] done (%.0fs)" % (time.time()-t0), flush=True)

def cpos(o):
    p = np.empty(len(o), int); p[o] = np.arange(len(o)); return p
def sfc(aff): return cpos(np.asarray(FQ.curve_order(aff)))
def lap_embed(Wm, k=3):
    Wm = np.array(Wm, float); np.fill_diagonal(Wm, 0.0); dd = Wm.sum(1)+1e-12; DM = 1.0/np.sqrt(dd)
    Ws = DM[:, None]*Wm*DM[None, :]; v, V = np.linalg.eigh(Ws); return V[:, list(range(-2, -2-k, -1))]
Ech = lap_embed(chan_aff, 3); pair = ['(%d,%d)' % (l, m) for l, m in chans]
def maxcorr(lab): return max(abs(np.corrcoef(Ech[:, d], lab)[0, 1]) for d in range(3))
rL, rAM, rNL, rM = maxcorr(larr), maxcorr(aml), maxcorr(nlat), maxcorr(marr)
print("[eastNN] channel |corr|: l=%.3f ; |m|=%.3f ; l-|m|=%.3f ; signed m=%.3f" % (rL, rAM, rNL, rM), flush=True)
np.savez_compressed(os.path.join(HERE, "quest3d_eastNN_data.npz"), X=X.astype(np.float32), row_aff=row_aff,
                    chan_aff=chan_aff, larr=larr, marr=marr, Ech=Ech, jstar=jstar)

norm08 = BoundaryNorm(np.arange(-0.5, Lmax+1.5, 1.0), Lmax+1); norm17 = BoundaryNorm(np.arange(0.5, Lmax+1.5, 1.0), Lmax)
cmdeg = ListedColormap(mpl.colormaps['Set1'].colors[:Lmax]); cpol = ListedColormap(mpl.colormaps['tab10'].colors[:Lmax+1]); cazi = ListedColormap(mpl.colormaps['Dark2'].colors[:Lmax+1])
fig = plt.figure(figsize=(19, 17))
axs = fig.add_subplot(2, 2, 1, projection='3d')
so = np.asarray(FQ.curve_order(row_aff))
axs.plot(X[so, 0], X[so, 1], X[so, 2], color='0.5', lw=.5, alpha=.55, zorder=1)
axs.scatter(X[:, 0], X[:, 1], X[:, 2], c=cpos(so), s=11, cmap='turbo', depthshade=False, zorder=3)
axs.view_init(elev=22, azim=45); axs.set_xticklabels([]); axs.set_yticklabels([]); axs.set_zticklabels([])
axs.set_title(r'Sphere $X$ space-filling curve (data-driven east pole)', fontsize=14)
def p3(pos, c, cm, nm, tk, title):
    ax = fig.add_subplot(2, 2, pos, projection='3d')
    sc = ax.scatter(Ech[:, 0], Ech[:, 1], Ech[:, 2], c=c, s=48, cmap=cm, norm=nm, alpha=.98, depthshade=False, edgecolors='k', linewidths=.4)
    for i in range(NC): ax.text(Ech[i, 0], Ech[i, 1], Ech[i, 2], pair[i], fontsize=6, ha='center', va='center')
    ax.view_init(elev=65, azim=324); ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([]); ax.set_title(title, fontsize=13)
    fig.colorbar(sc, ax=ax, shrink=.6, pad=.02, ticks=list(tk))
p3(2, larr, cmdeg, norm17, range(1, Lmax+1), r'by degree $\ell$ ($|r|=%.2f$)' % rL)
p3(3, nlat, cpol, norm08, range(0, Lmax+1), r'\textbf{by }$\bm{\ell-|m|}$\textbf{ parallels} ($\bm{|r|=%.2f}$)' % rNL)
p3(4, aml, cazi, norm08, range(0, Lmax+1), r'by $|m|$ meridians ($|r|=%.2f$)' % rAM)
fig.suptitle(r'$X{=}Y$ 3D questionnaire, \textbf{data-driven east pole} $j^*(i)=\arg\min_{j}|\langle x_i,x_j\rangle|$ ($N=%d$, $\ell\le%d$, %d iters)' % (N, Lmax, NIT), fontsize=14, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(HERE, 'quest3d_eastNN.png'); fig.savefig(out, dpi=140, bbox_inches='tight'); print("SAVED", out)
print("done (%.0fs)" % (time.time()-t0), flush=True)
