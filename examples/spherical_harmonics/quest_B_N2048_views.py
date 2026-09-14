"""10 tilted views of the N=2048 version-B (l,m) channel embedding, colored by l-|m| (parallels).
Replot from quest_B_N2048_data.npz."""
import os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import BoundaryNorm, ListedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa
matplotlib.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'text.latex.preamble': r'\usepackage{amsmath,amssymb}'})
HERE = os.path.dirname(os.path.abspath(__file__)); Lmax = 7
z = np.load(os.path.join(HERE, "quest_B_N2048_data.npz"))
E = z['Ech'].astype(float); larr = z['larr']; marr = z['marr']; nlat = larr - np.abs(marr)
lo = E.min(0); hi = E.max(0); pad = 0.05 * (hi - lo)
cmap = ListedColormap(mpl.colormaps['tab10'].colors[:Lmax + 1]); norm = BoundaryNorm(np.arange(-0.5, Lmax + 1.5, 1.0), Lmax + 1)
# tilts around the base view (elev 65, azim 324)
angles = [(65, 300), (65, 324), (65, 348), (58, 312), (58, 336), (72, 312), (72, 336), (50, 324), (80, 324), (65, 12)]
fig = plt.figure(figsize=(21, 8.6)); sc = None
for k, (el, az) in enumerate(angles):
    ax = fig.add_subplot(2, 5, k + 1, projection='3d')
    sc = ax.scatter(E[:, 0], E[:, 1], E[:, 2], c=nlat, s=42, cmap=cmap, norm=norm, alpha=.97, depthshade=False, edgecolors='k', linewidths=.3)
    ax.view_init(elev=el, azim=az)
    ax.set_xlim(lo[0]-pad[0], hi[0]+pad[0]); ax.set_ylim(lo[1]-pad[1], hi[1]+pad[1]); ax.set_zlim(lo[2]-pad[2], hi[2]+pad[2])
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.set_title(r'\#%d: elev$=%d^\circ$, azim$=%d^\circ$' % (k + 1, el, az), fontsize=10)
cb = fig.colorbar(sc, ax=fig.axes, shrink=.5, pad=.02, ticks=range(0, Lmax + 1)); cb.set_label(r'$\ell-|m|$ (parallels)', fontsize=12)
fig.suptitle(r'Version B, $N=2048$: $(\ell,m)$ channel embedding --- 10 tilted views, colored by $\ell-|m|$', fontsize=14, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(os.path.join(HERE, 'quest_B_N2048_views.png'), dpi=135, bbox_inches='tight')
fig.savefig(os.path.join(HERE, 'quest_B_N2048_views.pdf'), bbox_inches='tight'); print("SAVED")
