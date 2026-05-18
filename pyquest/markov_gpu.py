"""
markov.py: Function to calculate vectors and eigenvalues of Markov chain
           based on an affinity.
"""

import numpy as np
import scipy.sparse.linalg as spsl
import scipy.sparse as sps
from scipy.sparse.linalg import ArpackNoConvergence
import cupy as cp
from cupyx.scipy import sparse as cp_sps
from cupyx.scipy.sparse import linalg as cpsl

#### sparse matrices untested  #####

def make_markov_symmetric(data_gpu, thres=1e-8):
    """
    Pure GPU version of symmetric Markov normalization.
    Assumes data_gpu is already a cupy.ndarray or cupyx.scipy.sparse matrix.
    """
    # 1. Handle Sparse Data on GPU
    if cp_sps.issparse(data_gpu):
        # Filter by threshold
        d_mat = data_gpu.multiply(data_gpu > thres)
        m, n = d_mat.shape
        
        # First Normalization
        rowsums = 1.0 / (d_mat.sum(axis=1) + 1e-15)
        invD = cp_sps.spdiags(rowsums.T, 0, m, n)
        p_mat = invD * d_mat * invD
        
        # Second (Symmetric) Normalization
        rowsums_sqrt = cp.sqrt(1.0 / (p_mat.sum(axis=1) + 1e-15))
        sqrtInvD = cp_sps.spdiags(rowsums_sqrt.T, 0, m, n)
        return sqrtInvD * d_mat * sqrtInvD

    # 2. Handle Dense Data on GPU (Optimized for A4500)
    else:
        # Zero out elements below threshold using GPU kernel fusion
        d_mat = data_gpu * (data_gpu > thres)
        
        # First normalization using broadcasting
        # rowsums becomes a column vector (N, 1)
        rowsums = cp.sum(d_mat, axis=1) + 1e-15
        #p_mat = d_mat / (rowsums[:, cp.newaxis] * rowsums[cp.newaxis, :])
        inv_rowsums = 1.0 / rowsums
        p_mat = d_mat * inv_rowsums[:, cp.newaxis] * inv_rowsums[cp.newaxis, :]
        # Second normalization (Symmetric)
        d_mat2 = cp.sqrt(cp.sum(p_mat, axis=1)) + 1e-15
        # Final result stays in VRAM
        return p_mat / (d_mat2[:, cp.newaxis] * d_mat2[cp.newaxis, :])
        

def make_markov_row_stoch(data,thres=1e-8):
    """
    data is a (symmetric) affinity matrix. elements less than thres are zeroed.
    Returns the row stochastic Markov matrix. 
    """    
    
    
    if sps.issparse(data) == True:
        d_mat = data.multiply(data > thres)
        m,n =  np.shape(d_mat)
        rowsums = 1.0/(d_mat.sum(axis=1) + 1e-15)
        invD = sps.spdiags(rowsums.T, 0, m,n)
        p_mat = invD * d_mat
    else:
        d_mat = data*(data > thres)
        rowsums = 1.0/(np.sum(d_mat,axis=1) + 1e-15)
        p_mat = np.diag(rowsums).dot(d_mat)
    return p_mat

def markov_eigs(data,n_eigs,normalize=True,thres=1e-8):
    """
    data is a (symmetric) affinity matrix.
    n_eigs is the number of eigenvalues/eigenvectors desired.
    normalize sets whether to normalize all the eigenvectors such that the first
    eigenvector is 1.   (this function is ncut from the MATLAB questionnaire)
    Returns the first n eigenvectors and the corresponding eigenvalues.
    """
    p_mat = make_markov_symmetric(data,thres)
    return _calc_eigs(p_mat,n_eigs,normalize)

def _calc_eigs(markov_chain_gpu, n_eigs, normalize=True):
    """
    Optimized for Dense Symmetric Markov Chains on GPU.
    """
    n = markov_chain_gpu.shape[0]
    n_eigs = min(n_eigs, n)

    # 1. Use the specialized Symmetric Solver
    # eigh is much faster than svd for symmetric matrices
    eigenvalues, eigenvectors = cp.linalg.eigh(markov_chain_gpu)

    # 2. Sort Descending (eigh returns ascending)
    # Markov chain eigenvalues: largest is 1.0, we want the top n_eigs
    idx = cp.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx[:n_eigs]]
    eigenvectors = eigenvectors[:, idx[:n_eigs]]

    # 3. Normalization (In-place on GPU)
    if normalize:
        # Stationary distribution normalization (Vectorized)
        # Avoids slow hstack/tile logic
        eigenvectors /= eigenvectors[:, 0:1]

        # Orientation consistency (Vectorized)
        signs = cp.sign(eigenvectors[0, 1:])
        signs[signs == 0] = 1.0
        eigenvectors[:, 1:] *= signs[cp.newaxis, :]

    return eigenvectors, eigenvalues
    