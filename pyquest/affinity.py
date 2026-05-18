"""
affinity.py: Some basic routines for calculating initial affinities.
"""

import numpy as np
import scipy.spatial as spsp
import warnings
import numpy as np
import cupy as cp
from sklearn.metrics.pairwise import cosine_similarity as cosine_similarity_fast
from sklearn.random_projection import GaussianRandomProjection

def mutual_cosine_similarity(data, take_abs=False, no_data_value=None, threshold=0.0):
    """
    Fast version of cosine similarity between columns of data.
    - Ignores rows with missing values (np.nan or no_data_value).
    - Uses sklearn cosine_similarity for speed.
    """
    if no_data_value is not None:
        data = np.where(data == no_data_value, np.nan, data)

    # Identify rows with any missing value
    mask = np.isnan(data)
    valid_rows = ~np.any(mask, axis=1)

    # Keep only rows with complete data
    data_clean = data[valid_rows, :]  # shape: m_clean × n

    # Transpose to (n, m_clean) for cosine similarity between columns
    sim = cosine_similarity_fast(data_clean.T)

    if not take_abs:
        sim = np.maximum(sim, 0.0)

    if threshold > 0.0:
        sim[sim < threshold] = 0.0

    return sim




def mutual_cosine_similarity_gpu(data_gpu, take_abs=False, no_data_value=None, threshold=0.0):
    """
    Pure GPU version of mutual cosine similarity for columns.
    Assumes data_gpu is already a cupy.ndarray.
    """
    # 1. Handle missing values
    if no_data_value is not None:
        data_gpu = cp.where(data_gpu == no_data_value, cp.nan, data_gpu)

    # 2. Filter rows with NaNs (on GPU)
    # mask is True where NaNs exist
    mask = cp.isnan(data_gpu)
    # valid_rows is True only if a row has NO NaNs
    valid_rows = ~cp.any(mask, axis=1)
    data_clean = data_gpu[valid_rows, :]  # Shape: (m_clean, n)

    # 3. Compute Cosine Similarity via Matrix Multiplication
    # Formula: (A · B) / (||A|| * ||B||)
    
    # Calculate L2 norms for each column (axis=0)
    norms = cp.linalg.norm(data_clean, axis=0)
    
    # Prevent division by zero: if norm is 0, set to 1 to avoid Inf/NaN
    norms = cp.where(norms == 0, 1.0, norms)
    
    # Normalize columns: This turns the dot product into cosine similarity
    # Broadcasting happens entirely in GPU VRAM
    data_norm = data_clean / norms

    # THE BOTTLE NECK: Column-wise Dot Product
    # (n x m_clean) @ (m_clean x n) -> (n x n)
    # This triggers cuBLAS and the A4500's Tensor Cores
    sim = data_norm.T @ data_norm

    # 4. Post-processing (In-place on GPU)
    if take_abs:
        cp.abs(sim, out=sim)
    else:
        # np.maximum equivalent in cupy
        cp.maximum(sim, 0.0, out=sim)

    if threshold > 0.0:
        # Vectorized thresholding on GPU
        sim[sim < threshold] = 0.0

    return sim

def _norm_ip_abs_aff(data):
    """
    data: mxn numpy array.
    
    Treats columns of data as functions on R^m. values are expected to be 1/-1.
    Calculates affinity between columns of data as absolute value of inner
    product between columns divided by number of entries non-zero in both 
    columns.
    Returns nxn symmetric matrix of affinities on [0,1]
    """
    
    inner_products = data.T.dot(data)
    temp = np.abs(data)
    norm_constants = temp.T.dot(temp) + 1e-14
    
    return np.abs(inner_products) / norm_constants

def mutual_cosine_similarity0(data, take_abs=False, no_data_value=None,
                             threshold=0.0):
    """
    data: mxn numpy array.
    
    Treats columns of data as functions on R^m. 
    Calculates affinity between columns of data as cosine similarity between 
    columns, floored at 0 if take_abs == False, otherwise the absolute value. 
    If either vector has a value == no_data_value, then that dimension is 
    ignored. Automatically treats np.nan as no_data_value.
    Returns nxn symmetric matrix of affinities on [0,1]
    """
    
    mask = np.logical_or(np.isnan(data),data==no_data_value)
    madata = np.ma.MaskedArray(data,mask)
    
    valid = np.ones(np.shape(data)) * np.logical_not(madata.mask)
     
    inner_products = np.ma.dot(madata.T,madata)
    madata = madata**2
    ji_norm = np.ma.dot(valid.T,madata)
    ij_norm = np.ma.dot(madata.T,valid)
    mcs = inner_products/np.sqrt(ij_norm*ji_norm)
    
    if take_abs:
        return np.abs(np.array(mcs))
    else:
        if threshold is None:
            return np.array(mcs)
        else:
            mcs[mcs < threshold] = 0.0
            return np.array(mcs)

def cosine_similarity(data, take_abs=False):
    """
    data: mxn numpy array.
    
    Treats columns of data as functions on R^m. 
    Calculates affinity between columns of data as cosine similarity between 
    columns, floored at 0 if take_abs == False, otherwise the absolute value. 
    Counts 0 as a data value.
    Returns nxn symmetric matrix of affinities on [0,1]
    """

    norms = np.sqrt(np.sum(data**2,axis=0))
    inner_products = (data/norms).T.dot(data/norms)
    
    if take_abs:
        return np.abs(inner_products)
    else:
        inner_products[inner_products < 0.0] = 0.0
        return inner_products

def correlation(data, take_abs=False):
    """
    data: mxn numpy array.
    
    Treats columns of data as functions on R^m. values are expected to be 1/-1.
    Calculates affinity between columns of data as cosine similarity between 
    columns, floored at 0. Counts 0 as a data value.
    Returns nxn symmetric matrix of affinities on [0,1]
    """

    data2 = remove_mean(data)
    return cosine_similarity(data2,take_abs)

def remove_mean(data):
    """
    data : mxn numpy array
    Returns data, with the mean of each column subtracted.
    """
    means = np.mean(data,axis=0)
    return data - means

def gaussian_euclidean(data,knn=5,eps=1.0, sym=True):
    """
    data: mxn numpy array.
    
    Treats columns of data as functions on R^m. 
    Calculates affinity between columns of data as a Gaussian kernel of
    width eps*(median distance between 5 nearest neighbors of all points). 
    Returns nxn symmetric matrix of non-negative affinities.
    """
    import sklearn.neighbors as sknn

    row_distances = spsp.distance.squareform(spsp.distance.pdist(data.T))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nn = sknn.NearestNeighbors(n_neighbors=knn)
        nn.fit(data.T)
        dists,_ = nn.kneighbors(data.T,knn,True)
    medians = eps*np.median(dists,1) #axis=None
    medians[medians == 0] = 1
    mat = np.exp(-(row_distances**2/(medians**2)))
    if sym: # fixed bug: making affinity symmetric 
        mat = (mat+mat.T)/2
    return mat




def sparse_gaussian_euclidean(data,knn=5,eps=1.0):
	# sparse affinity untested #
    import sklearn.neighbors as sknn
    import scipy as sp
    
    D = sknn.kneighbors_graph(data.T, n_neighbors=knn, mode='distance')
    medians = eps * np.median(D.data)
    
    vals = np.exp(-(D.data**2/(medians**2)))
    mat = sp.sparse.csr_matrix( (vals,D.indices,D.indptr), np.shape(D) )
    return (mat+mat.T)/2

def threshold(affinity,threshold):
    """
    Takes an affinity and thresholds it by setting all entries to 0.0 which
    are less than threshold.
    """
    affinity[affinity < threshold] = 0.0
    return affinity        


def mutual_cosine_similarity_rp(data, take_abs=False, no_data_value=None, threshold=0.0):
    """
    Fast version of cosine similarity between columns of data.
    - Ignores rows with missing values (np.nan or no_data_value).
    - Uses sklearn cosine_similarity for speed.
    """
    if no_data_value is not None:
        data = np.where(data == no_data_value, np.nan, data)

    # Identify rows with any missing value
    mask = np.isnan(data)
    valid_rows = ~np.any(mask, axis=1)

    # Keep only rows with complete data
    data_clean = data[valid_rows, :]  # shape: m_clean × n
    
    # Transpose to (n, m_clean) for cosine similarity between columns
    sim = cosine_similarity_rp(data_clean,take_abs)

    if not take_abs:
        sim = np.maximum(sim, 0.0)

    if threshold > 0.0:
        sim[sim < threshold] = 0.0

    return sim

def cosine_similarity_rp(data, take_abs=False):
    """
    data: mxn numpy array.
    
    Treats columns of data as functions on R^m. 
    Calculates affinity between columns of data as cosine similarity between 
    columns, floored at 0 if take_abs == False, otherwise the absolute value. 
    Counts 0 as a data value.
    Returns nxn symmetric matrix of affinities on [0,1]
    """
    m,n = data.shape
    proj = GaussianRandomProjection(n_components=int(np.floor(m**(1/2))))
    X_proj = proj.fit_transform(data.T).T
    norms = np.sqrt(np.sum(X_proj**2,axis=0))
    inner_products = (X_proj/norms).T.dot(X_proj/norms)
    
    if take_abs:
        return np.abs(inner_products)
    else:
        inner_products[inner_products < 0.0] = 0.0
        return inner_products