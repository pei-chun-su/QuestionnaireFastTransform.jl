"""12 viewing directions of the 99-harmonic version-B channel embedding, colored by |m| (meridians).
Replot from quest_B_L9_data.npz."""
import os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import BoundaryNorm, ListedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa
matplotlib.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'text.latex.preamble': r'\usepackage{amsmath,amssymb}'})
HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(HERE, "quest_B_L9_data.npz"))
E = z['Ech'].astype(float); larr = z['larr']; marr = z['marr']; aml = np.abs(marr); Lmax = int(larr.max())
lo = E.min(0); hi = E.max(0); pad = 0.05 * (hi - lo)
big = ListedColormap((list(mpl.colormaps['tab10'].colors) + list(mpl.colormaps['Set1'].colors))[:Lmax + 1])
norm = BoundaryNorm(np.arange(-0.5, Lmax + 1.5, 1.0), Lmax + 1)
angles = [(el, az) for el in (25, 50, 75) for az in (0, 90, 180, 270)]     # 3 elevations x 4 azimuths = 12
fig = plt.figure(figsize=(21, 15)); sc = None
for k, (el, az) in enumerate(angles):
    ax = fig.add_subplot(3, 4, k + 1, projection='3d')
    sc = ax.scatter(E[:, 0], E[:, 1], E[:, 2], c=aml, s=34, cmap=big, norm=norm, alpha=.97, depthshade=False, edgecolors='k', linewidths=.3)
    ax.view_init(elev=el, azim=az)
    ax.set_xlim(lo[0]-pad[0], hi[0]+pad[0]); ax.set_ylim(lo[1]-pad[1], hi[1]+pad[1]); ax.set_zlim(lo[2]-pad[2], hi[2]+pad[2])
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.set_title(r'elev$=%d^\circ$, azim$=%d^\circ$' % (el, az), fontsize=11)
cb = fig.colorbar(sc, ax=fig.axes, shrink=.4, pad=.02, ticks=range(0, Lmax + 1)); cb.set_label(r'$|m|$ (meridians)', fontsize=12)
fig.suptitle(r'Version B, $N=2048$, 99 harmonics: $(\ell,m)$ channel embedding from 12 directions, colored by $|m|$ ($|r|=0.90$)', fontsize=14, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(os.path.join(HERE, 'quest_B_L9_views.png'), dpi=135, bbox_inches='tight')
fig.savefig(os.path.join(HERE, 'quest_B_L9_views.pdf'), bbox_inches='tight'); print("SAVED")
