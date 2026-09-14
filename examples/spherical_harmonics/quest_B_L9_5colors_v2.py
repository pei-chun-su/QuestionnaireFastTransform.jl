"""3D channel embedding (psi2,psi3,psi4) of the 99-harmonic version-B affinity, colored 5 ways:
degree l, signed m, |m| (meridians), l-|m| (parallels), and space-filling curve order.
Replot from quest_B_L9_data.npz."""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pyquest"))
import flip_questionnaire as FQ
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import BoundaryNorm, ListedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa
matplotlib.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'text.latex.preamble': r'\usepackage{amsmath,amssymb}'})
HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(HERE, "quest_B_L9_data.npz"))
E = z['Ech'].astype(float); larr = z['larr']; marr = z['marr']; chan_aff = np.asarray(z['chan_aff'])
aml = np.abs(marr); nlat = larr - aml; NC = len(larr); Lmax = int(larr.max())
def cpos(o):
    p = np.empty(len(o), int); p[o] = np.arange(len(o)); return p
chsfc = cpos(np.asarray(FQ.curve_order(chan_aff)))
mc = lambda lab: max(abs(np.corrcoef(E[:, d], lab)[0, 1]) for d in range(3))
pair = ['(%d,%d)' % (l, m) for l, m in zip(larr, marr)]
big = ListedColormap((list(mpl.colormaps['tab10'].colors) + list(mpl.colormaps['Set1'].colors))[:Lmax + 1])
n08 = BoundaryNorm(np.arange(-0.5, Lmax + 1.5, 1.0), Lmax + 1); n19 = BoundaryNorm(np.arange(0.5, Lmax + 1.5, 1.0), Lmax)

panels = [(larr, big, n19, range(1, Lmax + 1), r'degree $\ell$ ($|r|=%.2f$)' % mc(larr)),
          (marr, 'coolwarm', None, None, r'signed $m$ ($|r|=%.2f$)' % mc(marr)),
          (aml, big, n08, range(0, Lmax + 1), r'$|m|$ meridians ($|r|=%.2f$)' % mc(aml)),
          (nlat, big, n08, range(0, Lmax + 1), r'$\ell-|m|$ parallels ($|r|=%.2f$)' % mc(nlat)),
          (chsfc, 'turbo', None, None, r'space-filling curve order')]
fig = plt.figure(figsize=(21, 12.5))
for i, (c, cm, nm, tk, title) in enumerate(panels):
    ax = fig.add_subplot(2, 3, i + 1, projection='3d')
    kw = dict(s=34, cmap=cm, alpha=.97, depthshade=False, edgecolors='k', linewidths=.3)
    if nm is not None: kw['norm'] = nm
    elif cm == 'coolwarm': kw.update(vmin=-Lmax, vmax=Lmax)
    sc = ax.scatter(E[:, 0], E[:, 1], E[:, 2], c=c, **kw)
    for j in range(NC): ax.text(E[j, 0], E[j, 1], E[j, 2], pair[j], fontsize=4.4, ha='center', va='center')
    ax.view_init(elev=25, azim=90); ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([]); ax.set_title(title, fontsize=13)
    cb = fig.colorbar(sc, ax=ax, shrink=.6, pad=.02, **({'ticks': list(tk)} if tk is not None else {}))
fig.suptitle(r'Version B, $N=2048$, 99 harmonics ($\ell\le%d$): 3D channel embedding $(\psi_2,\psi_3,\psi_4)$ (view: elev $25^\circ$, azim $90^\circ$) colored by $\ell$, $m$, $|m|$, $\ell-|m|$, SFC' % Lmax, fontsize=14, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(HERE, 'quest_B_L9_5colors_v2.png'), dpi=140, bbox_inches='tight')
fig.savefig(os.path.join(HERE, 'quest_B_L9_5colors_v2.pdf'), bbox_inches='tight'); print("SAVED")
