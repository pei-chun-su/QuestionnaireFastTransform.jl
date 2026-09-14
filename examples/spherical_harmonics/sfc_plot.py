"""Sphere space-filling curve from the 2D questionnaire on X.X^T (version B). Loads compare_3d_vs_2d_data.npz."""
import os, sys, numpy as np
import flip_questionnaire as FQ
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
matplotlib.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'text.latex.preamble': r'\usepackage{amsmath,amssymb}'})
HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(HERE, "compare_3d_vs_2d_data.npz"))
X = z['X'].astype(float); row_aff = np.asarray(z['row_affB']); N = len(X)
def cpos(o):
    p = np.empty(len(o), int); p[o] = np.arange(len(o)); return p
so = np.asarray(FQ.curve_order(row_aff))           # space-filling curve order (leaf order of the X.X tree)
c = cpos(so)
angles = [(20, 40), (65, 324), (90, 0)]            # side, the #30-ish view, top-down
fig = plt.figure(figsize=(19, 6.6))
for k, (el, az) in enumerate(angles):
    ax = fig.add_subplot(1, 3, k + 1, projection='3d')
    ax.plot(X[so, 0], X[so, 1], X[so, 2], color='0.5', lw=.6, alpha=.6, zorder=1)
    sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=c, s=16, cmap='turbo', depthshade=False, zorder=3)
    ax.view_init(elev=el, azim=az); ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.set_xlabel(r'$x$'); ax.set_ylabel(r'$y$'); ax.set_zlabel(r'$z$')
    ax.set_title(r'elev$=%d^\circ$, azim$=%d^\circ$' % (el, az), fontsize=13)
cb = fig.colorbar(sc, ax=fig.axes, shrink=.5, pad=.02); cb.set_label('curve order', fontsize=12)
fig.suptitle(r'Sphere $X$ space-filling curve learned by the 2D questionnaire on $X\!\cdot\!X^\top$ ($N=%d$): points colored by curve order, curve drawn through them' % N, fontsize=14, y=1.0)
fig.savefig(os.path.join(HERE, 'sfc_plot.png'), dpi=140, bbox_inches='tight')
fig.savefig(os.path.join(HERE, 'sfc_plot.pdf'), bbox_inches='tight'); print("SAVED", "N=%d" % N)
