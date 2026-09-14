"""t-SNE of the (l,m) channel affinity (99 harmonics, version B), colored by degree l, |m| (meridians),
and l-|m| (parallels). Precomputed distance = -log(channel affinity) (proportional to the tree-EMD).
Reuses quest_B_L9_data.npz."""
import os, sys, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import BoundaryNorm, ListedColormap
from sklearn.manifold import TSNE
HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(HERE, "quest_B_L9_data.npz"))
W = np.asarray(z['chan_aff'], float); larr = z['larr']; marr = z['marr']
aml = np.abs(marr); nlat = larr - aml; NC = len(larr); Lmax = int(larr.max())
matplotlib.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'text.latex.preamble': r'\usepackage{amsmath,amssymb}'})
# affinity -> distance (proportional to the tree-EMD); symmetric, zero diagonal
W = 0.5 * (W + W.T); Wn = W / W.max(); np.fill_diagonal(Wn, 1.0)
D = -np.log(np.clip(Wn, 1e-9, 1.0)); D = 0.5 * (D + D.T); np.fill_diagonal(D, 0.0)
Y = TSNE(n_components=2, metric='precomputed', init='random', perplexity=15, random_state=0).fit_transform(D)
print("tSNE done, %d points" % NC, flush=True)

pair = ['(%d,%d)' % (l, m) for l, m in zip(larr, marr)]
big = ListedColormap((list(mpl.colormaps['tab10'].colors) + list(mpl.colormaps['Set1'].colors))[:Lmax + 1])
n08 = BoundaryNorm(np.arange(-0.5, Lmax + 1.5, 1.0), Lmax + 1); n19 = BoundaryNorm(np.arange(0.5, Lmax + 1.5, 1.0), Lmax)
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
specs = [(larr, big, n19, range(1, Lmax + 1), r'degree $\ell$'),
         (aml, big, n08, range(0, Lmax + 1), r'$|m|$ (meridians)'),
         (nlat, big, n08, range(0, Lmax + 1), r'$\ell-|m|$ (parallels)')]
for ax, (c, cm, nm, tk, lab) in zip(axes, specs):
    sc = ax.scatter(Y[:, 0], Y[:, 1], c=c, s=90, cmap=cm, norm=nm, edgecolors='k', linewidths=.4)
    for i in range(NC): ax.text(Y[i, 0], Y[i, 1], pair[i], fontsize=4.6, ha='center', va='center', color='k')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(r'colored by %s' % lab, fontsize=14)
    fig.colorbar(sc, ax=ax, shrink=.82, pad=.02, ticks=list(tk)).set_label(lab, fontsize=11)
fig.suptitle(r't-SNE of the $(\ell,m)$ channel affinity (99 harmonics, version B $N=2048$): perplexity 15, distance $=-\log$(affinity)', fontsize=14, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(HERE, 'tsne_channels.png'), dpi=140, bbox_inches='tight')
fig.savefig(os.path.join(HERE, 'tsne_channels.pdf'), bbox_inches='tight'); print("SAVED")
