"""Lay the (l,m) channels on the harmonic lattice (x=l, y=m), color each node by its position along the
learned channel space-filling curve, and draw the curve's path through the lattice. Two panels:
  A = 3D-questionnaire channel SFC,  B = 2D-on-X.X-then-EMD channel SFC.  From compare_3d_vs_2d_data.npz."""
import os, sys, numpy as np
import flip_questionnaire as FQ
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
matplotlib.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'text.latex.preamble': r'\usepackage{amsmath,amssymb}'})
HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(HERE, "compare_3d_vs_2d_data.npz"))
chanA = np.asarray(z['chanA']); chanB = np.asarray(z['chanB']); larr = z['larr']; marr = z['marr']; NC = len(larr)
def cpos(o):
    p = np.empty(len(o), int); p[o] = np.arange(len(o)); return p
def sfc(aff): return np.asarray(FQ.curve_order(aff))          # channel order along the space-filling curve

fig, axes = plt.subplots(1, 2, figsize=(18, 8.2))
for ax, aff, tag in [(axes[0], chanA, r'\textbf{A. 3D questionnaire} channel SFC'),
                     (axes[1], chanB, r'\textbf{B. 2D on }$X\!\cdot\!X$\textbf{ then EMD} channel SFC')]:
    order = sfc(aff); pos = cpos(order)
    ax.plot(larr[order], marr[order], '-', color='0.6', lw=1.2, alpha=.7, zorder=1)   # the curve's path
    sc = ax.scatter(larr, marr, c=pos, s=230, cmap='turbo', edgecolors='k', linewidths=.6, zorder=3)
    for i in range(NC):
        ax.text(larr[i], marr[i], '%d' % pos[i], fontsize=6.5, ha='center', va='center', color='k', zorder=4)
    ax.set_xlabel(r'degree $\ell$', fontsize=13); ax.set_ylabel(r'order $m$', fontsize=13)
    ax.set_xticks(range(1, 8)); ax.set_yticks(range(-7, 8)); ax.set_title(tag, fontsize=14)
    ax.grid(True, color='0.9', lw=.6); ax.set_axisbelow(True)
    cb = fig.colorbar(sc, ax=ax, shrink=.8, pad=.02); cb.set_label('space-filling curve order', fontsize=11)
fig.suptitle(r'Channel space-filling curve on the harmonic lattice $(\ell,m)$: node color \& number $=$ position along the curve, grey line $=$ its path', fontsize=14, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(HERE, 'sfc_on_lattice.png'), dpi=140, bbox_inches='tight')
fig.savefig(os.path.join(HERE, 'sfc_on_lattice.pdf'), bbox_inches='tight'); print("SAVED")
