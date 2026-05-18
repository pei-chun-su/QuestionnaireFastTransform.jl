"""
affinity.py: Some basic routines for calculating initial affinities.
"""

import numpy as np
import scipy.spatial as spsp
import warnings
import numpy as np
import cupy as cp
import aff_util2 as aff_util
from sklearn.metrics.pairwise import cosine_similarity as cosine_similarity_fast
from sklearn.random_projection import GaussianRandomProjection
from math import ceil
import multiprocessing as mp
import gc
import torch


def mutual_cosine_similarity_tiled_parallel(data_cpu, take_abs=False, no_data_value=None, threshold=0.0, tile_size=1000, num_gpus=8):
    # 1. Cleaning
    if no_data_value is not None:
        data_cpu[data_cpu == no_data_value] = np.nan
        
    mask = np.isnan(data_cpu)
    valid_rows = ~np.any(mask, axis=1)
    data_clean = np.ascontiguousarray(data_cpu[valid_rows, :], dtype=np.float32)
    
    n_cols = data_clean.shape[1]
    affinity = np.zeros((n_cols, n_cols), dtype=np.float32)
    
    # 2. Setup Task Queue
    task_width = 1000 
    args = []
    task_counter = 0
    for c_start in range(0, n_cols, task_width):
        c_end = min(c_start + task_width, n_cols)
        gpu_id = task_counter % num_gpus
        args.append((gpu_id, data_clean, c_start, c_end, tile_size, take_abs, threshold))
        task_counter += 1

    # 3. Execution (Spawn method is required for CUDA)
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    print(f"Distributing {task_counter} tasks across {num_gpus} GPUs using PyTorch...")
    
    with mp.Pool(processes=num_gpus, maxtasksperchild=5) as pool:
        # Note: Change 'aff_util' to your actual module name if this is in a separate file
        results = pool.starmap(aff_util.cosine_tile_worker, args)
        
    # 4. Assembly
    for res in results:
        if isinstance(res, tuple) and len(res) == 3:
            c_start, c_end, slice_data = res
            affinity[c_start:c_end, :] = slice_data
        else:
            print(f"Error on GPU {res[0]}: {res[1]}")

    # 5. In-place Symmetrization
    print("Finalizing symmetry...")
    affinity = (affinity + affinity.T) / 2.0
    
    return affinity

    

def mutual_cosine_similarity(data_gpu, take_abs=False, no_data_value=None, threshold=0.0):
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

def gaussian_euclidean_tiled_parallel(data_cpu, knn=5, eps=1.0, tile_size=2000, num_gpus=8):
    n_samples = data_cpu.shape[0]
    
    # 1. Prepare tasks
    args_pass1 = []
    for r_start in range(0, n_samples, tile_size):
        r_end = min(r_start + tile_size, n_samples)
        gpu_id = (r_start // tile_size) % num_gpus
        args_pass1.append((gpu_id, data_cpu, r_start, r_end, knn, None))

    # 2. Pass 1: Get KNN distances to compute Medians
    print(f"Pass 1: Computing KNN on {num_gpus} GPUs...")
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError: pass

    with mp.Pool(processes=num_gpus) as pool:
        knn_results = pool.starmap(aff_util.gaussian_tile_worker, args_pass1)

    # Reconstruct medians
    all_medians = np.zeros(n_samples)
    for r_start, r_end, knn_dists in knn_results:
        all_medians[r_start:r_end] = eps * np.median(knn_dists, axis=1)

    # 3. Pass 2: Compute Gaussian Matrix
    print(f"Pass 2: Computing Gaussian Affinity...")
    args_pass2 = []
    for r_start in range(0, n_samples, tile_size):
        r_end = min(r_start + tile_size, n_samples)
        gpu_id = (r_start // tile_size) % num_gpus
        args_pass2.append((gpu_id, data_cpu, r_start, r_end, knn, all_medians))

    affinity = np.zeros((n_samples, n_samples), dtype=np.float32)
    with mp.Pool(processes=num_gpus) as pool:
        final_results = pool.starmap(aff_util.gaussian_tile_worker, args_pass2)

    # 4. Assembly
    for r_start, r_end, slice_data in final_results:
        affinity[r_start:r_end, :] = slice_data

    # 5. Symmetrize
    print("Finalizing symmetry...")
    affinity = (affinity + affinity.T) / 2.0
    
    return affinity

def gaussian_euclidean(data, knn=5, eps=1.0, sym=True):
    """
    Pure GPU version of Gaussian Affinity.
    data: (features x samples) CuPy array.
    """
    # 1. Calculate Pairwise Squared Euclidean Distance on GPU
    # Using the identity: ||u-v||^2 = ||u||^2 + ||v||^2 - 2<u,v>
    data_t = data.T  # Samples are rows for distance calculation
    dot_product = data_t @ data_t.T
    sq_norms = cp.diag(dot_product)
    
    # Broadcasted distance calculation (Fastest way on A4500)
    dist_sq = sq_norms[:, cp.newaxis] + sq_norms[cp.newaxis, :] - 2 * dot_product
    dist_sq = cp.maximum(dist_sq, 0) # Remove tiny negatives from rounding
    
    # 2. Find K-Nearest Neighbors on GPU
    # Use partition to find the smallest distances without sorting the whole row
    # dist_sq has shape (n, n)
    knn_dists_sq = cp.partition(dist_sq, knn, axis=1)[:, :knn]
    knn_dists = cp.sqrt(knn_dists_sq)
    
    # 3. Calculate Medians for Kernel Width
    medians = eps * cp.median(knn_dists, axis=1)
    medians[medians == 0] = 1e-15 # Avoid division by zero
    
    # 4. Compute Gaussian Kernel
    # (dist_sq / medians^2) with broadcasting
    # Note: medians is (n,), we need (n, 1) for row-wise scaling
    mat = cp.exp(-(dist_sq / (medians[:, cp.newaxis]**2)))
    
    if sym:
        mat = (mat + mat.T) / 2.0
        
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

