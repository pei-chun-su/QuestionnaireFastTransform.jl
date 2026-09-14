"""Distribution of the east-pole minimum: for each x_i, m_i = min_{j!=i} |<x_i,x_j>| (0 = perfectly
orthogonal east pole). Shown as |cos| and as angular deviation from 90 deg, for several N."""
import os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
matplotlib.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'text.latex.preamble': r'\usepackage{amsmath,amssymb}'})
HERE = os.path.dirname(os.path.abspath(__file__))
Ns = [512, 1024, 2048, 4096]; cols = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
data = {}
for N in Ns:
    rng = np.random.default_rng(10); X = rng.standard_normal((N, 3)); X /= np.linalg.norm(X, axis=1, keepdims=True)
    C = np.abs(np.clip(X @ X.T, -1, 1)); np.fill_diagonal(C, np.inf)
    mn = C.min(1)                                                      # per-point east-pole |cos|
    data[N] = mn
    print("N=%4d : mean=%.5f  median=%.5f  max=%.4f  (deg dev: mean=%.4f, max=%.3f)" %
          (N, mn.mean(), np.median(mn), mn.max(), np.degrees(np.arcsin(mn)).mean(), np.degrees(np.arcsin(mn)).max()), flush=True)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.6))
for N, c in zip(Ns, cols):
    mn = data[N]
    a1.hist(mn, bins=60, histtype='step', density=True, color=c, lw=1.8, label=r'$N=%d$ (mean %.4f)' % (N, mn.mean()))
    a2.hist(np.degrees(np.arcsin(mn)), bins=60, histtype='step', density=True, color=c, lw=1.8, label=r'$N=%d$' % N)
a1.set_xlabel(r'east-pole minimum $\min_j|\langle x_i,x_j\rangle|$ ($0=$ orthogonal)', fontsize=12)
a1.set_ylabel('density', fontsize=12); a1.set_title(r'distribution of the east-pole $|\cos|$', fontsize=13); a1.legend(fontsize=10)
a2.set_xlabel(r'angular deviation from $90^\circ$ (degrees)', fontsize=12); a2.set_ylabel('density', fontsize=12)
a2.set_title(r'east pole: how far from perfectly orthogonal', fontsize=13); a2.legend(fontsize=10)
fig.suptitle(r'Data-driven east pole $j^*(i)=\arg\min_{j}|\langle x_i,x_j\rangle|$: how close the nearest-orthogonal sphere point is, vs.\ $N$', fontsize=13, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(HERE, 'east_pole_dist.png'), dpi=145, bbox_inches='tight')
fig.savefig(os.path.join(HERE, 'east_pole_dist.pdf'), bbox_inches='tight'); print("SAVED")
