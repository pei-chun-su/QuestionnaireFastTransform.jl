"""
markov.py: Function to calculate vectors and eigenvalues of Markov chain
           based on an affinity.
"""

import numpy as np
import scipy.sparse.linalg as spsl
import scipy.sparse as sps
from scipy.sparse.linalg import ArpackNoConvergence
from sklearn.utils.extmath import randomized_svd

#### sparse matrices untested  #####

def make_markov_symmetric(data,thres=1e-8):
    """
    data is a (symmetric) affinity matrix. elements less than thres are zeroed.
    Returns a "symmetrized" and normalized version of the Markov chain matrix. 
    """    
    
    
    if sps.issparse(data) == True:
        d_mat = data.multiply(data > thres)
        rowsums = 1.0/(d_mat.sum(axis=1) + 1e-15)
        m,n =  np.shape(d_mat)
        invD = sps.spdiags(rowsums.T, 0,m,n)
        p_mat = invD * d_mat * invD
        rowsums = np.sqrt(1.0/(p_mat.sum(axis=1) + 1e-15))
        sqrtInvD = sps.spdiags(rowsums.T, 0, m,n)
        p_mat = sqrtInvD * d_mat * sqrtInvD
    else:
        d_mat = data*(data > thres)
        rowsums = np.sum(d_mat,axis=1) + 1e-15
        p_mat = d_mat/(np.outer(rowsums,rowsums))
        d_mat2 = np.sqrt(np.sum(p_mat,axis=1)) + 1e-15
        p_mat = p_mat/(np.outer(d_mat2,d_mat2))
    return p_mat

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

def markov_eigs(data,n_eigs,normalize=True,thres=1e-8, diag_bias = False):
    """
    data is a (symmetric) affinity matrix.
    n_eigs is the number of eigenvalues/eigenvectors desired.
    normalize sets whether to normalize all the eigenvectors such that the first
    eigenvector is 1.   (this function is ncut from the MATLAB questionnaire)
    Returns the first n eigenvectors and the corresponding eigenvalues.
    """
    p_mat = make_markov_symmetric(data,thres)
    return _calc_eigs(p_mat,n_eigs,normalize, diag_bias)



def _calc_eigs(markov_chain, n_eigs, normalize=True, diag_bias = False):
    n = markov_chain.shape[0]
    n_eigs = min(n_eigs, n - 1)
    
    # 1. Sparsity handling
    if sps.issparse(markov_chain):
        if markov_chain.format != 'csr':
            markov_chain = markov_chain.tocsr()
    else:
        density = np.count_nonzero(markov_chain) / markov_chain.size
        if density < 0.1:
            markov_chain = sps.csr_matrix(markov_chain)

    # 2. Add Bias (Regularization to ensure stability)
    rng = np.random.default_rng(1357)
    if diag_bias:
        bias = rng.uniform(0, 1e-10, n)
        if sps.issparse(markov_chain):
            markov_chain = markov_chain + sps.diags(bias, format='csr')
        else:
            markov_chain = markov_chain + np.diag(bias)

    # 3. Solver Selection based on Matrix Size
    # Use Randomized SVD for N > 2^11 (2048)
    if n > 1000:
        print(f"Using Randomized SVD for N={n}")
        # n_iter=7 is the power iteration count. 
        u, s, vt = randomized_svd(markov_chain, 
                                  n_components=n_eigs,
                                  n_iter=7, 
                                  random_state=1357)
        eigenvalues = s
        eigenvectors = u
    else:
        dense_mat = markov_chain.toarray() if sps.issparse(markov_chain) else markov_chain
        eigenvalues, eigenvectors = np.linalg.eig(dense_mat)
    
    idx = np.argsort(-np.abs(eigenvalues))
    eigenvalues = eigenvalues[idx[:n_eigs]]
    eigenvectors = eigenvectors[:, idx[:n_eigs]]

    # 4. Post-processing
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    if normalize:
        # Stationary distribution normalization (first eigenvector)
        # We ensure no division by zero with a tiny epsilon
        denom = eigenvectors[:, [0]]
        denom[denom == 0] = 1e-12 
        eigenvectors /= denom
        
        # Consistent sign flipping based on the first element of each vector
        signs = np.sign(eigenvectors[0, 1:])
        signs[signs == 0] = 1.0
        eigenvectors[:, 1:] *= signs

    return eigenvectors, eigenvalues
"""
def _calc_eigs(markov_chain,n_eigs,normalize=True):
    n = np.shape(markov_chain)[0]
    n_eigs = min(n_eigs,n)
    rng = np.random.default_rng(1357) 
    bias = rng.uniform(0, 1e-10, n)
    if hasattr(markov_chain, "toarray"):
        from scipy import sparse
        markov_chain = markov_chain + sparse.diags(bias)
    else:
        markov_chain = markov_chain + np.diag(bias)
        
    try:
        # ARPACK requires k < n. If k == n, we must use the dense solver.
        if n_eigs >= n:
            raise ValueError("Dimensions too small for sparse solver")
            
        [vectors, singvals, _] = spsl.svds(markov_chain, n_eigs, 
                                           random_state=1357, 
                                           maxiter=max(2000, n*20)) # Increased maxiter helps convergence

    # 2. Fallback to dense solver if SVDS fails (Convergence Error or Dimension Error)
    except (spsl.ArpackNoConvergence, ValueError, RuntimeError):
        
        # Convert to dense matrix if necessary
        if hasattr(markov_chain, "toarray"):
            dense_mat = markov_chain.toarray()
        else:
            dense_mat = markov_chain
        u, s, vt = np.linalg.svd(dense_mat, full_matrices=False)
        
        # Slice only the ones we need
        vectors = u[:, :n_eigs]
        singvals = s[:n_eigs]

    y = np.argsort(-singvals)
    eigenvalues = singvals[y]
    eigenvectors = vectors[:, y]
    
    if normalize:
        n_mat = np.hstack([np.reshape([eigenvectors[:,0]],[-1,1])]*n_eigs)
        eigenvectors /= n_mat
        n_mat2 = np.vstack([np.sign(eigenvectors[0,1:])]*n)
        n_mat2[n_mat2==0] = 1.0
        eigenvectors[:,1:] *= n_mat2
        
    return eigenvectors, eigenvalues
"""