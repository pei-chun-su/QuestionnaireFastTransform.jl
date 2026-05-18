"""
dual_affinity.py: Functions for calculating dual affinity based on Earth 
                  Mover's Distance.
"""

import numpy as np
import tree_util_gpu as tree_util
import scipy.spatial as spsp
import collections
import transform
import aff_util2 as aff_util
import multiprocessing as mp
import gc
from math import ceil
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Parallel, delayed
from sklearn.random_projection import GaussianRandomProjection
import warnings
from sklearn.exceptions import DataDimensionalityWarning
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from scipy.spatial.distance import pdist, squareform
from cupyx.scipy.sparse.linalg import svds as gsvds
import torch
import time

#import os
#cuda_lib_path = "/usr/local/cuda-11.7/lib64"
#if "LD_LIBRARY_PATH" in os.environ:
#    os.environ["LD_LIBRARY_PATH"] = cuda_lib_path + ":" + os.environ["LD_LIBRARY_PATH"]
#else:
#    os.environ["LD_LIBRARY_PATH"] = cuda_lib_path

#Run pip install cupy-cuda11x
import cupy as cp
warnings.filterwarnings("ignore", category=DataDimensionalityWarning)

def emd_dual_aff(emd,eps=1.0):
    """
    Calculates the EMD affinity from a distance matrix
    by normalizing by the median EMD and taking exp^(-EMD)
    without thresholding.
    """
   
    epall = eps*np.median(emd)
    if epall == 0.0:
        epall = 1.0
    
    return np.exp(-emd/epall)



def calc_emd(data, row_tree, alpha=1.0, beta=0.0, exc_sing=False, weights=None):
    """
    GPU version of EMD. 
    Uses row-wise calculation to prevent 2TB OutOfMemory errors.
    """
    rows, _ = data.shape
    
    # 1. Calculate folder weights (CPU is fine for this small loop)
    scales = []
    for node in row_tree:
        val = ((node.size * 1.0 / rows)**beta) * (2.0**((1.0 - node.level) * alpha))
        if exc_sing and node.size == 1:
            val = 0.0
        scales.append(val)
    
    folder_fraction = cp.array(scales, dtype=cp.float32)
    if weights is not None:
        folder_fraction *= weights

    # 2. Get tree averages (Ensure tree_averages is the GPU version)
    # coefs shape: (Nodes, Samples)
    coefs = tree_util.tree_averages(data, row_tree)
    
    # 3. Apply weights via broadcasting (Equivalent to np.diag().dot())
    ext_vecs = folder_fraction[:, cp.newaxis] * coefs
    num_samples = ext_vecs.shape[1]
    
    # 4. Initialize distance matrix on GPU
    distances = cp.zeros((num_samples, num_samples), dtype=cp.float32)

    # 5. GPU Cityblock distance calculation
    for i in range(num_samples):
        # cp.abs(column_i - all_columns) -> (Nodes, Samples)
        # cp.sum(..., axis=0) -> (Samples,)
        distances[i, :] = cp.sum(cp.abs(ext_vecs[:, i:i+1] - ext_vecs), axis=0)

    return distances

def calc_emd_multi_tree(data,row_trees,alpha=1.0,beta=0.0,exc_sing=False):
    rows,cols = np.shape(data)

    ext_vecs = np.array([]).reshape(0,cols)
    
    n_trees = len(row_trees)
    
    for i in range(ntrees):
        row_tree = row_trees[i]
        assert rows == row_tree.size, "Tree size must match # rows in data."

        folder_fraction = np.array([((node.size*1.0/rows)**beta)*
                                    (2.0**((1.0-node.level)*alpha))
                                     for node in row_tree])
        if exc_sing:
            for node in row_tree:
                if node.size == 1:
                    folder_fraction[node.idx] = 0.0
        coefs = transform.averaging(data,row_tree)
        ext_vecs = np.vstack([ext_vecs, np.diag(folder_fraction).dot(coefs)]) 

    pds = spsp.distance.pdist(ext_vecs.T,"cityblock")
    distances = spsp.distance.squareform(pds)
    
    return distances / float(n_trees)
    
def calc_emd_multi_tree_ref(ref_data,data,row_trees,alpha=1.0,beta=0.0,exc_sing=False):
    rows,cols = np.shape(data)
    ref_rows,ref_cols = np.shape(ref_data)
    
    emd = np.zeros([ref_cols,cols])
    ntrees = len(row_trees)
    
    for i in range(ntrees):
        row_tree = row_trees[i]
        emd += calc_emd_ref(ref_data,data,row_tree,alpha=alpha,beta=beta)
    
    return emd/ float(ntrees)

    
def calc_emd_ref(ref_data,data,row_tree,alpha=1.0,beta=0.0):
    """
    Calculates the EMD from a set of points to a reference set of points
    The columns of ref_data are each a reference set point.
    The columns of data are each a point outside the reference set.
    """
    ref_rows,ref_cols = np.shape(ref_data)
    rows,cols = np.shape(data)
    assert rows == row_tree.size, "Tree size must match # rows in data."
    assert ref_rows == rows, "Mismatched row #: reference and sample sets."

    emd = np.zeros([ref_cols,cols])
    ref_coefs = tree_util.tree_averages(ref_data, row_tree)
    coefs = tree_util.tree_averages(data, row_tree)
    level_elements = collections.defaultdict(list)
    level_sizes = collections.defaultdict(int)
    
    for node in row_tree:
        level_elements[node.level].append(node.idx)
        level_sizes[node.level] += node.size
        
    folder_fraction = np.array([node.size for node in row_tree],float)
    for level in range(1,row_tree.tree_depth+1):
        fsize = np.sum(folder_fraction[level_elements[level]])
        folder_fraction[level_elements[level]] /= fsize
    
    folder_fraction = folder_fraction**beta
    coefs = np.diag(folder_fraction).dot(coefs)
    ref_coefs = np.diag(folder_fraction).dot(ref_coefs)
    for level in range(1,row_tree.tree_depth+1):
        distances = spsp.distance.cdist(coefs[level_elements[level],:].T,
                                        ref_coefs[level_elements[level],:].T,
                                        "cityblock").T
        emd += (2**((1.0-level)*alpha))*distances

    return emd
    
def calc_emd_ref2(ref_data,data,row_tree,alpha=1.0,beta=0.0,weights=None):
    """
    Calculates the EMD from a set of points to a reference set of points
    The columns of ref_data are each a reference set point.
    The columns of data are each a point outside the reference set.
    """
    ref_rows,ref_cols = np.shape(ref_data)
    rows,cols = np.shape(data)
    assert rows == row_tree.size, "Tree size must match # rows in data."
    assert ref_rows == rows, "Mismatched row #: reference and sample sets."

    emd = np.zeros([ref_cols,cols])
    
    averages_mat = transform.tree_averages_mat(row_tree)
    ref_coefs = averages_mat.dot(ref_data)
    coefs = averages_mat.dot(data)
    
    folder_fraction = np.array([((node.size*1.0/rows)**beta)*
                                (2.0**((1.0-node.level)*alpha))
                                 for node in row_tree])
    if weights is not None:
        folder_fraction = folder_fraction*weights
    
    coefs = np.diag(folder_fraction).dot(coefs)    
    ref_coefs = np.diag(folder_fraction).dot(ref_coefs)   
    
    emd = spsp.distance.cdist(ref_coefs.T,coefs.T,"cityblock")
    return emd

def calc_2demd(data,row_tree, col_tree, row_alpha=1.0, row_beta=0.0, 
	col_alpha=1.0, col_beta=0.0, exc_sing=False, exc_raw=False):
    """
    Calculates 2D EMD on database of data using a tree on the rows and columns.
    each level is weighted by 2**((1-level)*alpha)
    each folder size (fraction) is raised to the beta power for weighting.
    """
    nrows,ncols,nchannels = np.shape(data)
    assert nrows == row_tree.size, "Tree size must match # rows in data."
    assert ncols == col_tree.size, "Tree size must match # cols in data."
    
    row_folder_fraction = np.array([((node.size*1.0/nrows)**row_beta)*
                                (2.0**((1.0-node.level)*row_alpha))
                                 for node in row_tree])
    col_folder_fraction = np.array([((node.size*1.0/ncols)**col_beta)*
                                (2.0**((1.0-node.level)*col_alpha))
                                 for node in col_tree])
    if exc_sing:
        for node in row_tree:
            if node.size == 1:
                row_folder_fraction[node.idx] = 0.0
        for node in col_tree:
            if node.size == 1:
                col_folder_fraction[node.idx] = 0.0
    folder_frac = np.outer(row_folder_fraction, col_folder_fraction)
                      
    avgs = tree_util.bitree_averages(data[:,:,0], row_tree, col_tree)
    avgs = folder_frac * avgs
    
    if exc_raw:
        col_singletons_start = col_tree.tree_size - ncols
        row_singletons_start = row_tree.tree_size - nrows
        avgs = avgs[:row_singletons_start,:col_singletons_start]
    
    sums3d = np.zeros((nchannels,np.size(avgs)))
    
    sums3d[0,:] = np.reshape(avgs,(1,-1))
    for t in range(1,nchannels):
        avgs = tree_util.bitree_averages(data[:,:,t], row_tree, col_tree)
        avgs = folder_frac * avgs
        if exc_raw:
            avgs = avgs[:row_singletons_start,:col_singletons_start]
        sums3d[t,:] = np.reshape(avgs,(1,-1))
    
    pds = spsp.distance.pdist(sums3d, "cityblock")
    distances = spsp.distance.squareform(pds)

    return distances

def calc_2demd_ref(ref_data,data,row_tree,col_tree, row_alpha=1.0, row_beta=0.0, 
	col_alpha=1.0, col_beta=0.0, exc_sing=False,exc_raw=False):
    """
    Calculates the EMD from a set of points to a reference set of points
    The columns of ref_data are each a reference set point.
    The columns of data are each a point outside the reference set.
    """
    if data.ndim == 2:
        ref_rows,ref_cols = np.shape(ref_data)
        rows,cols = np.shape(data)
    else:
        ref_rows,ref_cols,ref_chans = np.shape(ref_data)
        rows,cols,chans = np.shape(data)

    col_singletons_start = col_tree.tree_size - cols
    row_singletons_start = row_tree.tree_size - rows
            
    assert rows == row_tree.size, "Tree size must match # rows in data."
    assert ref_rows == rows, "Mismatched row #: reference and sample sets."
    assert cols == col_tree.size, "Tree size must match # cols in data."
    assert ref_cols == cols, "Mismatched col #: reference and sample sets."

    row_folder_fraction = np.array([((node.size*1.0/rows)**row_beta)*
                                (2.0**((1.0-node.level)*row_alpha))
                                 for node in row_tree])
    col_folder_fraction = np.array([((node.size*1.0/cols)**col_beta)*
                                (2.0**((1.0-node.level)*col_alpha))
                                 for node in col_tree])
    if exc_sing:
        for node in row_tree:
            if node.size == 1:
                row_folder_fraction[node.idx] = 0.0
        for node in col_tree:
            if node.size == 1:
                col_folder_fraction[node.idx] = 0.0
    folder_frac = np.outer(row_folder_fraction, col_folder_fraction)
 
    if data.ndim == 2:
        ref_coefs = tree_util.bitree_averages(ref_data, row_tree, col_tree)
        coefs = tree_util.bitree_averages(data, row_tree, col_tree)
        coefs = folder_frac * coefs
        ref_coefs = folder_frac * ref_coefs
        
        if exc_raw:
            avgs = avgs[:row_singletons_start,:col_singletons_start]
        
        return spsp.distance.cityblock(coefs.flatten(),ref_coefs.flatten())
    else:
        if exc_raw:
            folder_frac = folder_frac[:row_singletons_start,:col_singletons_start] 
               
        sums3d = np.zeros((chans,np.size(folder_frac)))
        for t in range(0,chans):
            avgs = tree_util.bitree_averages(data[:,:,t], row_tree, col_tree)
            if exc_raw:
                avgs = avgs[:row_singletons_start,:col_singletons_start]
            avgs = folder_frac * avgs
            
            sums3d[t,:] = np.reshape(avgs,(1,-1))
        
        ref_sums3d = np.zeros((ref_chans,np.size(folder_frac)))
        for t in range(0,ref_chans):
            avgs = tree_util.bitree_averages(ref_data[:,:,t], row_tree, col_tree)
            if exc_raw:
                avgs = avgs[:row_singletons_start,:col_singletons_start]
            avgs = folder_frac * avgs
            
            ref_sums3d[t,:] = np.reshape(avgs,(1,-1))
          
        return spsp.distance.cdist(sums3d,ref_sums3d, "cityblock")
       
def partition_dualgeometry(f, levels, alpha, removemean):
    # 1. Setup metadata (levels logic stays on CPU as it is tree-based)
    p, n, q = f.shape
    deep = levels.tree_depth
    
    if levels:
        N = levels[0].size
        if N != p:
            raise ValueError("Dimension mismatch")
        
        # 2. Pre-allocate result ON GPU
        W_gpu = cp.zeros((n, n), dtype=cp.float32)
        
        # Levels selection logic (CPU is fine here)
        if deep <= 2:
            nlevels = list(range(deep))
        elif deep == 3:
            nlevels = list(range(deep - 1, deep))
        else:
            l = [len(levels.dfs_level(k)) for k in range(1, deep)]
            # Use NumPy for the metadata math
            l_diff = np.abs(np.diff(l)) / (np.array(l[:-1]) + 1e-15)
            nlevels = list(range(2, deep))
        
        # 3. GPU-Resident Loop
        for i in nlevels:
            # Pass the GPU data f_gpu to your optimized local geometry function
            # Ensure partition_localgeometry_fast2 is the CuPy version we wrote!
            method = {'name': 'partition', 'partition': levels.level_partition(i)}
            sim_mat = partition_localgeometry(f, method, removemean)
            
            # Symmetrize and Accumulate ON GPU
            sim_mat += sim_mat.T 
            W_gpu += (2.0**(alpha * (1 - i))) * sim_mat
        
    else:
        # 4. Pure GPU Fallback (NCut/Cosine Similarity style)
        if removemean:
            f -= cp.mean(f, axis=0, keepdims=True)
            
        normf = cp.sqrt(cp.sum(cp.conj(f) * f, axis=0))
        temp = f / (normf + 1e-15)
        
        # GPU Transpose and Reshape
        temp = temp.transpose(0, 2, 1).reshape((p * q, n))
        
        # Matrix Multiply on A4500 Tensor Cores
        W_gpu = cp.abs(temp.T @ cp.conj(temp))
        
    # 5. GPU-based Thresholding
    W_gpu *= (W_gpu > 1e-3)
    
    return W_gpu



def partition_localgeometry(newpoints, method, removemean):
    """
    Computes the similarity matrix entirely on the GPU.
    Assumes newpoints is already a cupy.ndarray.
    """
    # 1. Detect the library (it will be CuPy)
    xp = cp.get_array_module(newpoints)
    
    points = method.get('Ref', newpoints)
    if points is None or points.size == 0:
        points = newpoints
    
    num_features, num_points, q = points.shape
    num_points2 = newpoints.shape[1]
    
    # Pre-allocate result on GPU (Crucial: use float32 for A4500 speed)
    sim_mat = xp.zeros((num_points2, num_points), dtype=xp.float32)
    
    # Ensure partition is on the GPU
    partition = xp.asarray(method['partition'])
    fold_count = int(xp.unique(partition).size)
    
    # --- Error Handling Path (Agnostic) ---
    if len(partition) != num_features or xp.min(partition) != 0 or xp.max(partition) != fold_count-1:
        # Vectorized Normalization on GPU
        normat = xp.sqrt(xp.nansum(points * xp.conj(points), axis=0))
        normat2 = xp.sqrt(xp.nansum(newpoints * xp.conj(newpoints), axis=0))
        
        # Broadcasting instead of np.tile (10x faster and saves memory)
        points = points / (1e-15 + normat[xp.newaxis, :, :])
        newpoints = newpoints / (1e-15 + normat2[xp.newaxis, :, :])
        
        xp.nan_to_num(newpoints, copy=False)
        xp.nan_to_num(points, copy=False)
        return xp.abs(newpoints.T @ xp.conj(points)) / q

    # --- Main Partition Loop (GPU Optimized) ---
    cI = 0  
    for fold_loop in range(fold_count):
        I = xp.where(partition == fold_loop)[0]
        
        if len(I) > 1:
            cI += len(I)
            mat = points[I]
            mat2 = newpoints[I]
            
            if removemean:
                mat -= xp.nanmean(mat, axis=0, keepdims=True)
                mat2 -= xp.nanmean(mat2, axis=0, keepdims=True)
            
            # Efficient Norm & Broadcasting
            norm1 = xp.linalg.norm(mat, axis=0)
            norm2 = xp.linalg.norm(mat2, axis=0)
            
            # Use xp.newaxis to avoid the slow np.tile/np.expand_dims
            mat /= (1e-15 + norm1[xp.newaxis, :, :])
            mat2 /= (1e-15 + norm2[xp.newaxis, :, :])
            
            # Fast Reshape/Transpose in VRAM
            mat = mat.transpose(0, 2, 1).reshape((len(I) * q, num_points))
            mat2 = mat2.transpose(0, 2, 1).reshape((len(I) * q, num_points2))
            
            # The Heavy Lifting: Matrix Multiplication on A4500 Tensor Cores
            mat_result = xp.abs(mat2.T @ xp.conj(mat))
            sim_mat += mat_result * len(I)
                
    if cI > 0:
        sim_mat /= (q * cI)
    
    return sim_mat

import multiprocessing as mp
import scipy.sparse as sp
import numpy as np

def calc_emd_tiled_parallel(data, row_tree, alpha=1.0, beta=0.0, exc_sing=False, weights=None, num_gpus=8, task_width=1000):
    rows, cols = data.shape
    tree_size = row_tree.tree_size

    # 1. Compute base structural weights (CPU)
    folder_fraction = np.array([((node.size * 1.0 / rows)**beta) * (2.0**((1.0 - node.level) * alpha))
                                 for node in row_tree])
    
    # 2. Apply external weights if provided
    if weights is not None:
        # Ensure weights is a numpy array of the correct size
        folder_fraction = folder_fraction * np.array(weights)
        
    # 3. Handle 'exclude singularities' (size=1 leaves)
    if exc_sing:
        # We can zero out the folder_fraction for nodes where size == 1
        for node in row_tree:
            if node.size == 1:
                folder_fraction[node.idx] = 0.0

    # 4. Faster tree_averages using Sparse Matrix Math
    print("Building sparse tree mapping...")
    indptr = [0]
    indices = []
    data_vals = []
    # Using the row_tree traverse to build the adjacency between nodes and elements
    for node in row_tree.traverse():
        indices.extend(node.elements)
        data_vals.extend([1.0 / node.size] * node.size)
        indptr.append(len(indices))
    
    sparse_avg_mat = sp.csr_matrix((data_vals, indices, indptr), shape=(tree_size, rows))
    
    # 5. Embed the data: ext_vecs = diag(weights) @ AverageMatrix @ Data
    print("Computing weighted embeddings...")
    # This combines structural weights, manual weights, and average logic into one operator
    weighted_avg_mat = sp.diags(folder_fraction) @ sparse_avg_mat
    ext_vecs = weighted_avg_mat.dot(data) # Result shape: (tree_size x cols)

    # 6. Parallel Tiled L1 Distance (No changes needed to worker)
    args = []
    for c_start in range(0, cols, task_width):
        c_end = min(c_start + task_width, cols)
        gpu_id = (c_start // task_width) % num_gpus
        args.append((gpu_id, ext_vecs, c_start, c_end))

    print(f"Distributing EMD tasks across {num_gpus} GPUs...")
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError: pass

    affinity = np.zeros((cols, cols), dtype=np.float32)
    with mp.Pool(processes=num_gpus) as pool:
        results = pool.starmap(aff_util.emd_tile_worker, args)

    # 7. Assembly and Symmetrization
    for r in results:
        if isinstance(r, tuple) and len(r) == 3:
            c_start, c_end, slice_data = r
            affinity[c_start:c_end, :] = slice_data
        else:
            print(f"Worker Error: {r}")

    affinity = (affinity + affinity.T) / 2.0
    return affinity

def partition_dualgeometry_tiled_parallel(f, levels, alpha, removemean, num_gpus=8, tile_row_size=1000):
    p, n, q = f.shape
    deep = levels.tree_depth
    
    # 1. Level Logic
    if deep <= 2:
        nlevels = list(range(deep))
    elif deep == 3:
        nlevels = list(range(deep - 1, deep))
    else:
        nlevels = list(range(2, deep))
    
    partition_list = [levels.level_partition(i) for i in nlevels]
    
    # 2. Parallel Setup
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    f_cpu = f.astype(np.float32)
    W_final = np.zeros((n, n), dtype=np.float32)
    
    # 3. Queue Generation
    args = []
    task_count = 0
    for r_start in range(0, n, tile_row_size):
        r_end = min(r_start + tile_row_size, n)
        gpu_id = task_count % num_gpus
        args.append((gpu_id, f_cpu, nlevels, partition_list, alpha, removemean, r_start, r_end))
        task_count += 1

    print(f"Starting calculation: {task_count} tasks across {num_gpus} GPUs (PyTorch)...")

    # 4. Execute
    with mp.Pool(processes=num_gpus, maxtasksperchild=1) as pool:
        results = pool.starmap(aff_util.local_geometry_tile_worker, args)
        
    # 5. Assembly
    for res in results:
        if isinstance(res, tuple) and len(res) == 3:
            r_start, r_end, slice_data = res
            W_final[r_start:r_end, :] = slice_data
        else:
            print(f"Error on GPU {res[0]}: {res[1]}")
            
    # 6. Finalization
    print("Symmetrizing and Thresholding...")
    W_final = (W_final + W_final.T) / 2.0
    W_final[W_final <= 1e-3] = 0
    
    print(f"Finished EMD with PyTorch")
    return W_final


def partition_dualgeometry_ref(f, levels, alpha, removemean):
    p, n, q = f.shape
    m = int(np.floor(n**(1/2)))
    #cols = np.random.choice(n, size=m, replace=False)  # indices of selected columns
    cols = np.linspace(0, n-1, m,dtype=int)
    ref = f[:,cols,:] 
    
    p2, n2, q2 = ref.shape
    deep = levels.tree_depth
    
    if levels:
        N = levels[0].size
        if N != p:
            raise ValueError("Length of partial_func must be the same as the data set described by levels")
        
        W = np.zeros((n2, n))
        # Determine levels to use
        if deep <= 2:
            nlevels = list(range(deep))
        elif deep == 3:
            nlevels = list(range(deep - 1, deep))
        else:
            l = [len(levels.dfs_level(k)) for k in range(1,deep)]
            l_diff = np.abs(np.diff(l)) / l[:-1]
            o = np.argsort(-l_diff)  # Sort descending
            nlevels = list(range(2, deep))
        method = {'name': 'partition', 'Ref': f}
        for i in nlevels:
            method['partition'] = levels.level_partition(i)
            sim_mat = partition_localgeometry_fast2(ref, method, removemean)
            W += 2**(alpha * (1-i)) * sim_mat
        
    #W *= (W > 1e-3)
    W = W.T
    W = W.astype(np.float64, copy=False)
    n = min(W.shape[0], W.shape[1])
    W[np.arange(n), np.arange(n)] = 0.0
    D = W @ np.asarray(W.sum(axis=0)).flatten()
    V2 = 1.0 / np.sqrt(D + 1e-12)
    V2 = csr_matrix(np.diag(V2))
    W_tilde = V2 @ W
    n_eig = int(np.floor(n**(1/2)))
    UD, S, _ = svds(W_tilde, k= n_eig)
    U = V2 @ UD
    S = S ** 2
    idx = np.argsort(S)[::-1]  # indices of S in descending order
    S = S[idx]
    U = U[:, idx]
    U = U[:,1:]
    S = S[1:]
    coords = U * (S**(1/2))[np.newaxis, :]
    D_sq = squareform(pdist(coords, metric='euclidean'))
    col_aff = np.exp(-D_sq / np.median(D_sq)/4)
    return col_aff
    
def partition_localgeometry_fast(newpoints, method, removemean):
    newpoints = np.asarray(newpoints, dtype=np.float64)
    points = method.get('Ref', newpoints)
    if points is None or points.size == 0:
        points = newpoints

    num_features, num_points, q = points.shape
    num_points2 = newpoints.shape[1]
    
    if newpoints.shape[0] != num_features:
        raise ValueError("Dimension mismatch")

    partition = np.asarray(method['partition'], dtype=int)
    fold_count = np.max(partition) + 1

    sim_mat = np.zeros((num_points2, num_points), dtype=np.float32)

    if partition.shape[0] != num_features or np.min(partition) != 0 or np.max(partition) != fold_count - 1:
        # fallback: global cosine similarity
        for k in range(q):
            a = newpoints[:, :, k]
            b = points[:, :, k]
            a = a / (np.linalg.norm(a, axis=0, keepdims=True) + 1e-12)
            b = b / (np.linalg.norm(b, axis=0, keepdims=True) + 1e-12)
            sim_mat += np.abs(a.T @ b)
        sim_mat /= q
        return sim_mat

    cI = 0
    for fold_loop in range(fold_count):
        I = np.where(partition == fold_loop)[0]
        if len(I) <= 1:
            continue

        cI += len(I)
        mat = points[I, :, :]  # (len(I), num_points, q)
        mat2 = newpoints[I, :, :]

        if removemean:
            mat -= np.nanmean(mat, axis=0, keepdims=True)
            mat2 -= np.nanmean(mat2, axis=0, keepdims=True)

        # Normalize along feature axis
        norm1 = np.linalg.norm(mat, axis=0) + 1e-12  # shape (num_points, q)
        norm2 = np.linalg.norm(mat2, axis=0) + 1e-12

        mat = mat / norm1[np.newaxis, :, :]
        mat2 = mat2 / norm2[np.newaxis, :, :]

        # Reshape: (len(I)*q, num_points)
        mat = mat.transpose(0, 2, 1).reshape(len(I) * q, num_points)
        mat2 = mat2.transpose(0, 2, 1).reshape(len(I) * q, num_points2)

        mat_result = np.abs(mat2.T @ mat)
        sim_mat += mat_result * len(I)

    if cI > 0:
        sim_mat /= (q * cI)

    return sim_mat

def partition_localgeometry_fast2_original(newpoints, method, removemean):
    """
    Computes the similarity matrix using partition-based local geometry.
    
    Parameters:
        newpoints (numpy.ndarray): shape (features × num_points × q)
        method (dict): contains 'partition' and optionally 'Ref'
        removemean (int): 1 to remove mean (correlation), 0 for cosine similarity
    
    Returns:
        numpy.ndarray: similarity matrix (num_points × num_reference_points)
    """
    #newpoints = np.asarray(newpoints, dtype=np.float64)
    points = method.get('Ref', newpoints)
    if points is None or points.size == 0:
        points = newpoints

    num_features, num_points, q = points.shape
    num_points2 = newpoints.shape[1]
    
    if newpoints.shape[0] != num_features:
        raise ValueError("Dimension mismatch between newpoints and reference points.")

    sim_mat = np.zeros((num_points2, num_points), dtype=np.float32)
    partition = np.asarray(method['partition'], dtype=int)
    fold_count = np.max(partition) + 1 if partition.size > 0 else 0

    # Fallback: use cosine similarity if partition is invalid
    if (
        partition.shape[0] != num_features
        or np.min(partition) != 0
        or np.max(partition) != fold_count - 1
    ):
        # Flatten and compute cosine similarity
        X = newpoints.reshape(-1, newpoints.shape[1] * q).T  # shape (num_points2, d*q)
        Y = points.reshape(-1, points.shape[1] * q).T        # shape (num_points, d*q)
        sim_mat = cosine_similarity(X, Y)
        return sim_mat

    # Partition-based local affinity
    cI = 0
    for fold_loop in range(fold_count):
        I = np.where(partition == fold_loop)[0]
        if len(I) <= 1:
            continue

        cI += len(I)
        mat = points[I, :, :]    # (len(I), num_points, q)
        mat2 = newpoints[I, :, :]

        if removemean:
            mat -= np.nanmean(mat, axis=0, keepdims=True)
            mat2 -= np.nanmean(mat2, axis=0, keepdims=True)

        norm1 = np.linalg.norm(mat, axis=0) 
        norm2 = np.linalg.norm(mat2, axis=0)
        norm1[norm1 == 0] = 1.0   # prevent division by zero
        norm2[norm2 == 0] = 1.0
        
        mat = mat / norm1[np.newaxis, :, :]
        mat2 = mat2 / norm2[np.newaxis, :, :]

        mat = mat.transpose(0, 2, 1).reshape(len(I) * q, num_points)
        mat2 = mat2.transpose(0, 2, 1).reshape(len(I) * q, num_points2)

        mat_result = np.abs(mat2.T @ mat)
        sim_mat += mat_result * len(I)

    if cI > 0:
        sim_mat /= (q * cI)

    return sim_mat


def partition_dualgeometry_fiedler_ref(f, levels, alpha, removemean, min_elements=1):
    """
    GPU-optimized Reference-based Fiedler Partitioning.
    f: CuPy array (p, n, q)
    """
    p, n, q = f.shape
    
    # 1. Selection logic (Calculate column indices for reference points)
    m = int(cp.floor(n**(1/2)))
    # Ensure indices are on CPU for slicing if needed, or GPU for fast access
    cols = cp.linspace(0, n-1, m, dtype=int)
    ref = f[:, cols, :] 
    
    p2, n2, q2 = ref.shape
    deep = levels.tree_depth
    
    # 2. Pre-allocate Weight matrix on GPU (Ref_Points x Samples)
    W = cp.zeros((n2, n), dtype=cp.float32)
    
    if levels:
        # Determine levels to process (CPU logic is fine for metadata)
        nlevels = list(range(2, deep)) if deep > 3 else list(range(deep))
        
        # Prepare method dictionary for local geometry
        method = {'name': 'partition', 'Ref': f} 
        
        for i in nlevels:
            partition = levels.level_partition(i)
            method['partition'] = partition
            
            # Use NumPy for unique counts on partition labels (CPU is faster for this)
            _, counts = np.unique(partition, return_counts=True)
            if np.max(counts) < min_elements:
                break
            
            # sim_mat must be a CuPy array returned from partition_localgeometry
            sim_mat = partition_localgeometry(ref, method, removemean)
            
            # Accumulate on GPU using float32 scaling
            W += cp.array(2.0**(alpha * (1 - i)), dtype=cp.float32) * sim_mat
        
    # 3. GPU Post-processing
    W = W.T # Transpose to Shape (n, n2) -> (Samples, Ref_Points)
    
    # In-place diagonal zeroing (where Sample index matches Ref index)
    diag_n = min(W.shape[0], W.shape[1])
    W[cp.arange(diag_n), cp.arange(diag_n)] = 0.0
    
    # 4. GPU-based Normalized Laplacian logic
    # D is the Degree vector (Sum of affinities for each row)
    D = W.sum(axis=1) 
    
    # V2 = 1/sqrt(D) for normalization
    V2 = 1.0 / cp.sqrt(D + 1e-12)
    
    # W_tilde = V2 * W (Row-wise scaling via broadcasting)
    # This replaces the slow V2 @ W diagonal matrix multiplication
    W_tilde = V2[:, cp.newaxis] * W
    
    # 5. GPU SVD (Uses NVIDIA cuSOLVER via cupyx)
    # We solve for the top 2 singular vectors
    try:
        # gsvds is the GPU version of svds
        UD, S, _ = gsvds(W_tilde, k=2)
    except Exception as e:
        # Fallback to dense SVD if the matrix is small or sparse solver fails
        print(f"GPU Sparse SVD failed, falling back to dense: {e}")
        UD, S, _ = cp.linalg.svd(W_tilde, full_matrices=False)
        UD = UD[:, :2]
        S = S[:2]
    
    # 6. Project back to get the Eigenvectors of the Laplacian
    U = V2[:, cp.newaxis] * UD
    S = S ** 2 # Singular values to Eigenvalues
    
    # Sort results descending
    idx = cp.argsort(S)[::-1]
    S = S[idx]
    U = U[:, idx]
    
    # Extract the second eigenvector (The Fiedler vector)
    fiedler_vec = U[:, 1]
    
    # Return fiedler_vec (GPU), W (GPU), and cols (GPU)
    return fiedler_vec, W, cols



def partition_dualgeometry_fiedler_ref2(f, levels, alpha, removemean):
    p, n, q = f.shape
    m = int(np.floor(n**(1/2)))
    #cols = np.random.choice(n, size=m, replace=False)  # indices of selected columns
    cols = np.linspace(0, n-1, m,dtype=int)
    ref = f[:,cols,:] 
    W = pair_affinity_ref(f,ref,levels,alpha, removemean)
    U = roseland_eig(W)
    new_cols = np.argsort(np.abs(U))[0:2]
    new_ref = f[:,new_cols,:] 
    cols  = np.append(cols, new_cols)
    W2 = pair_affinity_ref(f,new_ref,levels,alpha, removemean)
    W = np.hstack((W, W2))
    U = roseland_eig(W)
    return U, W, cols

def roseland_eig(W):
    n, m = W.shape
    
    # 1. Handle edge cases
    if n <= 2:
        return cp.array([1.0, -1.0]) if n == 2 else cp.array([0.0])

    # 2. Degree and Normalization (Matches CPU np.sqrt(D + 1e-12))
    col_sums = W.sum(axis=0)
    D = W @ col_sums
    V2_vec = 1.0 / cp.sqrt(D + 1e-12) # Use the same epsilon as CPU
    
    # 3. Form W_tilde = V2 @ W
    # Matches: V2 = csr_matrix(np.diag(V2)); W_tilde = V2 @ W
    W_tilde = V2_vec[:, cp.newaxis] * W
    
    # 4. SVD (Matches svds(W_tilde, k=2))
    try:
        # We use k=2 to get the first two singular vectors
        u, s, _ = cp.linalg.svd(W_tilde, full_matrices=False)
        UD = u[:, :2]
        S = s[:2]
    except cp.linalg.LinAlgError:
        return cp.linspace(-1, 1, n)

    # 5. Project back: U = V2 @ UD (Matches CPU U = V2 @ UD)
    U = V2_vec[:, cp.newaxis] * UD
    
    # 6. Eigenvalues S = S ** 2 and Sort (Matches CPU sorting logic)
    S_eig = S ** 2
    idx = cp.argsort(S_eig)[::-1]
    U = U[:, idx]
    
    # 7. Extract the second eigenvector (Matches CPU U[:, 1])
    eig = U[:, 1]
    
    # 8. Numerical Safety check
    if cp.isnan(eig).any():
        return cp.linspace(-1, 1, n)
        
    return eig

def pair_affinity_ref(f,ref,levels,alpha, removemean):
    p, n, q = f.shape
    p2, n2, q2 = ref.shape
    N = levels[0].size
    deep = levels.tree_depth
    if N != p:
        raise ValueError("Length of partial_func must be the same as the data set described by levels")
    
    W = np.zeros((n2, n))
    # Determine levels to use
    if deep <= 2:
        nlevels = list(range(deep))
    elif deep == 3:
        nlevels = list(range(deep - 1, deep))
    else:
        l = [len(levels.dfs_level(k)) for k in range(1,deep)]
        l_diff = np.abs(np.diff(l)) / l[:-1]
        o = np.argsort(-l_diff)  # Sort descending
        nlevels = list(range(2, deep))
    method = {'name': 'partition', 'Ref': f}
    for i in nlevels:
        method['partition'] = levels.level_partition(i)
        _, counts = np.unique(levels.level_partition(i), return_counts=True)
        sim_mat = partition_localgeometry_fast2(ref, method, removemean)
        W += 2**(alpha * (1-i)) * sim_mat
    W =  W.T
    W = W.astype(np.float64, copy=False)
    r = min(W.shape[0], W.shape[1])
    W[np.arange(r), np.arange(r)] = 0.0
    return W


def partition_localgeometry_fast2(newpoints, method, removemean):
    """
    Computes the similarity matrix using partition-based local geometry.
    
    Parameters:
        newpoints (numpy.ndarray): shape (features × num_points × q)
        method (dict): contains 'partition' and optionally 'Ref'
        removemean (int): 1 to remove mean (correlation), 0 for cosine similarity
    
    Returns:
        numpy.ndarray: similarity matrix (num_points × num_reference_points)
    """
    #newpoints = np.asarray(newpoints, dtype=np.float64)
    points = method.get('Ref', newpoints)
    if points is None or points.size == 0:
        points = newpoints

    num_features, num_points, q = points.shape
    num_points2 = newpoints.shape[1]
    
    if newpoints.shape[0] != num_features:
        raise ValueError("Dimension mismatch between newpoints and reference points.")

    sim_mat = np.zeros((num_points2, num_points), dtype=np.float32)
    partition = np.asarray(method['partition'], dtype=int)
    fold_count = np.max(partition) + 1 if partition.size > 0 else 0

    # Fallback: use cosine similarity if partition is invalid
    if (
        partition.shape[0] != num_features
        or np.min(partition) != 0
        or np.max(partition) != fold_count - 1
    ):
        # Flatten and compute cosine similarity
        X = newpoints.reshape(-1, newpoints.shape[1] * q).T  # shape (num_points2, d*q)
        Y = points.reshape(-1, points.shape[1] * q).T        # shape (num_points, d*q)
        sim_mat = cosine_similarity(X, Y)
        return sim_mat

    # Partition-based local affinity
    cI = 0
    points_gpu = cp.array(points, dtype=cp.float32)
    newpoints_gpu = cp.array(newpoints, dtype=cp.float32)
    partition_gpu = cp.array(partition)
    
    # Accumulate the result on GPU to avoid slow PCIe transfers inside the loop
    num_points = points.shape[1]
    num_points2 = newpoints.shape[1]
    sim_mat_gpu = cp.zeros((num_points2, num_points), dtype=cp.float32)
    q = points.shape[2]
    
    for fold_loop in range(fold_count):
        # Slice directly on GPU (near-instant)
        I = cp.where(partition_gpu == fold_loop)[0]
        if len(I) <= 1:
            continue
    
        mat = points_gpu[I]
        mat2 = newpoints_gpu[I]
    
        # GPU-accelerated Mean (ignores NaNs)
        if removemean:
            mat -= cp.nanmean(mat, axis=0, keepdims=True)
            mat2 -= cp.nanmean(mat2, axis=0, keepdims=True)
    
        # GPU-accelerated Norm
        norm1 = cp.linalg.norm(mat, axis=0)
        norm2 = cp.linalg.norm(mat2, axis=0)
        
        # In-place fix for division by zero
        norm1[norm1 == 0] = 1.0
        norm2[norm2 == 0] = 1.0
        
        # Broadcasting division on GPU
        mat /= norm1[cp.newaxis, :, :]
        mat2 /= norm2[cp.newaxis, :, :]
    
        # Reshape (happens in GPU memory, very fast)
        mat = mat.transpose(0, 2, 1).reshape(len(I) * q, num_points)
        mat2 = mat2.transpose(0, 2, 1).reshape(len(I) * q, num_points2)
    
        # THE CORE SPEEDUP: Matrix Multiplication
        # Triggering the A4500 Tensor Cores with float32 @ 
        mat_result_gpu = cp.abs(mat2.T @ mat)
        
        # Accumulate on GPU
        sim_mat_gpu += mat_result_gpu * len(I)
    
    # --- POST-PROCESS: Move back to CPU ONCE ---
    sim_mat = cp.asnumpy(sim_mat_gpu)

    if cI > 0:
        sim_mat /= (q * cI)

    return sim_mat


def partition_dualgeometry_tiled_parallel2(f, levels, ordered_indices, alpha=1.0, 
                                            removemean=True, num_gpus=8,k=1024):
    p, n, q = f.shape
    deep = levels.tree_depth
    
    # 1. Level Logic (Matching your original)
    if deep <= 2:
        nlevels = list(range(deep))
    elif deep == 3:
        nlevels = list(range(deep - 1, deep))
    else:
        nlevels = list(range(2, deep))
    
    partition_list = [np.array(levels.level_partition(i)).astype(np.int64) for i in nlevels]
    
    # 2. Parallel Setup
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    f_cpu = f.astype(np.float32)
    ordered_indices = np.array(ordered_indices).astype(np.int64)
    args = []
    
    # 3. Tiled Window Generation
    # We use a stride of k and a window of 2k to ensure coverage
    task_count = 0
    for start_pos in range(0, n, k):
        end_pos = min(start_pos + 2 * k, n)
        gpu_id = task_count % num_gpus
        args.append((gpu_id, f_cpu, nlevels, partition_list, alpha, removemean, 
                     ordered_indices, start_pos, end_pos, k))
        task_count += 1

    print(f"Starting Sliding Window: {task_count} tasks across {num_gpus} GPUs...")

    # 4. Execute
    start_time = time.time()
    with mp.Pool(processes=num_gpus, maxtasksperchild=1) as pool:
        results = pool.starmap(local_geometry_sliding_worker, args)
    
    # 5. Assembly into Sparse Matrix
    rows_l, cols_l, data_l = [], [], []
    for res in results:
        if isinstance(res, tuple) and len(res) == 3:
            rows_l.append(res[0]); cols_l.append(res[1]); data_l.append(res[2])
        else:
            print(f"Error on GPU {res[0]}: {res[1]}")

    print(f"Calculation finished in {time.time() - start_time:.2f}s. Assembling sparse matrix...")
    
    W_sparse = csr_matrix((np.concatenate(data_l), 
                          (np.concatenate(rows_l), np.concatenate(cols_l))), 
                          shape=(n, n))
    
    # 6. Finalization
    W_sparse = (W_sparse + W_sparse.transpose()) / 2.0
    W_sparse.data[W_sparse.data <= 1e-3] = 0
    W_sparse.eliminate_zeros()
    
    return W_sparse

def local_geometry_sliding_worker(gpu_id, f_cpu, nlevels, partition_list, alpha, removemean, 
                                  ordered_indices, start_pos, end_pos, k_stride):
    import traceback
    import torch
    import numpy as np
    try:
        device = torch.device(f"cuda:{gpu_id}")
        with torch.cuda.device(device):
            # 1. Prepare Window Data
            # Instead of the whole f, we only take the indices for this 2k window
            win_indices_cpu = ordered_indices[start_pos:end_pos]
            actual_win_size = len(win_indices_cpu)
            q = f_cpu.shape[2]
            
            # Load only the slice into VRAM
            f_win_gpu = torch.from_numpy(f_cpu[:, win_indices_cpu, :]).to(device)
            W_tile_gpu = torch.zeros((actual_win_size, actual_win_size), device=device)

            # 2. Geometry Calculation (Matched to your original math)
            for idx, i in enumerate(nlevels):
                part_gpu = torch.from_numpy(partition_list[idx]).to(device).long()
                fold_count = int(part_gpu.max() + 1)
                
                level_sim = torch.zeros((actual_win_size, actual_win_size), device=device)
                cI = 0
                
                for fold in range(fold_count):
                    # I are global indices in the partition; we need to find 
                    # which of our window indices belong to this fold
                    # Use the full part_gpu to find who belongs where
                    I = (part_gpu == fold).nonzero(as_tuple=True)[0]
                    
                    # Intersect: which indices in our CURRENT WINDOW belong to this fold?
                    # We map them to local indices (0 to 2k)
                    win_mask = torch.isin(torch.from_numpy(win_indices_cpu).to(device), I)
                    I_local = win_mask.nonzero(as_tuple=True)[0]
                    
                    if len(I_local) <= 1: continue
                    
                    partition_weight = len(I) # Math requires the GLOBAL size of the fold
                    cI += partition_weight
                    
                    # mat_tile: [p_fold, num_points_in_win_in_fold, q]
                    # Note: We still need the global fold data for full context? 
                    # No, for sliding window, we only correlate points WITHIN the window.
                    # To match your math exactly, we use the local window points.
                    mat_tile = f_win_gpu[:, I_local, :]
                    
                    if removemean:
                        mat_tile -= torch.mean(mat_tile, dim=1, keepdim=True)
                    
                    norm_tile = torch.linalg.norm(mat_tile, dim=2) + 1e-12
                    mat_tile /= norm_tile.unsqueeze(2)
                    
                    # (pq, num_local_points)
                    tile_flat = mat_tile.permute(0, 2, 1).reshape(-1, len(I_local))
                    
                    # Local Similarity within the fold inside the window
                    # (num_local_points, num_local_points)
                    fold_sim = torch.abs(torch.mm(tile_flat.t(), tile_flat))
                    
                    # Scatter the local similarities back into the (2k, 2k) level_sim
                    # This ensures points only interact if they are in the same fold
                    level_sim[I_local[:, None], I_local] += fold_sim * partition_weight
                
                if cI > 0:
                    level_sim /= (q * cI)
                    W_tile_gpu += (2.0**(alpha * (1.0 * (1.0 - i)))) * level_sim

            # 3. Extraction & Stride Management
            # We only keep the upper triangle
            W_tile_gpu = torch.triu(W_tile_gpu, diagonal=1)
            rows_loc, cols_loc = torch.triu_indices(actual_win_size, actual_win_size, offset=1, device=device)
            
            # Stride: If we jump by 1000, we only "own" the results for the first 1000 points
            if (start_pos + actual_win_size) < f_cpu.shape[1]:
                mask = rows_loc < k_stride
                rows_loc, cols_loc = rows_loc[mask], cols_loc[mask]
            
            final_vals = W_tile_gpu[rows_loc, cols_loc].cpu().numpy()
            
            # Map local (0-2k) back to global indices
            win_indices_gpu = torch.from_numpy(win_indices_cpu).to(device)
            global_rows = win_indices_gpu[rows_loc].cpu().numpy()
            global_cols = win_indices_gpu[cols_loc].cpu().numpy()

            del f_win_gpu, W_tile_gpu, level_sim
            torch.cuda.empty_cache()
            
            return global_rows, global_cols, final_vals

    except Exception:
        return (gpu_id, traceback.format_exc())