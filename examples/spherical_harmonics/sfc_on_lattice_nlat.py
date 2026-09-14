"""(l,m) lattice (x=l, y=m) with nodes colored by l-|m| (parallels, categorical), the channel SFC path
drawn through them, and each node numbered by its SFC-order position. A = 3D questionnaire, B = 2D+EMD.
l-|m| is constant along lattice anti-diagonals, so this shows whether the SFC path follows those bands."""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pyquest"))
import flip_questionnaire as FQ
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import BoundaryNorm, ListedColormap
matplotlib.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'text.latex.preamble': r'\usepackage{amsmath,amssymb}'})
HERE = os.path.dirname(os.path.abspath(__file__)); Lmax = 7
z = np.load(os.path.join(HERE, "compare_3d_vs_2d_data.npz"))
chanA = np.asarray(z['chanA']); chanB = np.asarray(z['chanB']); larr = z['larr']; marr = z['marr']
nlat = larr - np.abs(marr); NC = len(larr)
def cpos(o):
    p = np.empty(len(o), int); p[o] = np.arange(len(o)); return p
def sfc(aff): return np.asarray(FQ.curve_order(aff))
cmap = ListedColormap(mpl.colormaps['tab10'].colors[:Lmax + 1]); norm = BoundaryNorm(np.arange(-0.5, Lmax + 1.5, 1.0), Lmax + 1)

fig, axes = plt.subplots(1, 2, figsize=(18, 8.2))
for ax, aff, tag in [(axes[0], chanA, r'\textbf{A. 3D questionnaire} channel SFC'),
                     (axes[1], chanB, r'\textbf{B. 2D on }$X\!\cdot\!X$\textbf{ then EMD} channel SFC')]:
    order = sfc(aff); pos = cpos(order)
    ax.plot(larr[order], marr[order], '-', color='0.55', lw=1.3, alpha=.75, zorder=1)     # SFC path
    sc = ax.scatter(larr, marr, c=nlat, s=250, cmap=cmap, norm=norm, edgecolors='k', linewidths=.6, zorder=3)
    for i in range(NC):
        ax.text(larr[i], marr[i], '%d' % pos[i], fontsize=6.5, ha='center', va='center', color='k', zorder=4)
    ax.set_xlabel(r'degree $\ell$', fontsize=13); ax.set_ylabel(r'order $m$', fontsize=13)
    ax.set_xticks(range(1, 8)); ax.set_yticks(range(-7, 8)); ax.set_title(tag, fontsize=14)
    ax.grid(True, color='0.9', lw=.6); ax.set_axisbelow(True)
    cb = fig.colorbar(sc, ax=ax, shrink=.8, pad=.02, ticks=range(0, Lmax + 1)); cb.set_label(r'$\ell-|m|$ (parallels)', fontsize=12)
fig.suptitle(r'Channel space-filling curve on the $(\ell,m)$ lattice, nodes colored by $\ell-|m|$ (parallels); number $=$ SFC-order position, grey line $=$ curve path', fontsize=14, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(HERE, 'sfc_on_lattice_nlat.png'), dpi=140, bbox_inches='tight')
fig.savefig(os.path.join(HERE, 'sfc_on_lattice_nlat.pdf'), bbox_inches='tight'); print("SAVED")
