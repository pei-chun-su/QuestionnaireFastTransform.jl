"""
dual_affinity.py: Functions for calculating dual affinity based on Earth 
                  Mover's Distance.
"""

import numpy as np
import tree_util
import scipy.spatial as spsp
import collections
import transform
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Parallel, delayed
from sklearn.random_projection import GaussianRandomProjection
import warnings
from sklearn.exceptions import DataDimensionalityWarning
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from scipy.spatial.distance import pdist, squareform

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

def calc_emd(data,row_tree,alpha=1.0,beta=0.0,exc_sing=False,weights=None):
    """
    Calculates the EMD on the *columns* from data and a tree on the rows.
    each level is weighted by 2**((1-level)*alpha)
    each folder size (fraction) is raised to the beta power for weighting.
    """
    rows,_ = np.shape(data)
    assert rows == row_tree.size, "Tree size must match # rows in data."

    folder_fraction = np.array([((node.size*1.0/rows)**beta)*
                                (2.0**((1.0-node.level)*alpha))
                                 for node in row_tree])
    if weights is not None:
        folder_fraction = folder_fraction*weights
    
    if exc_sing:
        for node in row_tree:
            if node.size == 1:
                folder_fraction[node.idx] = 0.0
    coefs = tree_util.tree_averages(data,row_tree)
    
    ext_vecs = np.diag(folder_fraction).dot(coefs)
    
    pds = spsp.distance.pdist(ext_vecs.T,"cityblock")
    distances = spsp.distance.squareform(pds)

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
       
def partition_dualgeometry(f, levels, alpha, removemean ):
    p, n, q = f.shape
    deep = levels.tree_depth
    
    if levels:
        N = levels[0].size
        if N != p:
            raise ValueError("Length of partial_func must be the same as the data set described by levels")
        
        W = np.zeros((n, n))
        
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
        
        for i in nlevels:
            sim_mat = partition_localgeometry_fast2(f, {'name': 'partition', 'partition': levels.level_partition(i)}, removemean)
            sim_mat = sim_mat + sim_mat.T  # Symmetrize
            W += 2**(alpha * (1-i)) * sim_mat
        
    else:
        f = f - np.mean(f, axis=0, keepdims=True)
        normf = np.sqrt(np.sum(np.conj(f) * f, axis=0))
        temp = f / (normf + np.finfo(float).eps)
        temp = temp.transpose(0, 2, 1).reshape((p * q, n))
        W = np.abs(temp.T @ np.conj(temp))
        
    diag = np.diag(W)
    inv_sqrt_diag = 1.0 / np.sqrt(diag + np.finfo(float).eps)
    W = W * np.outer(inv_sqrt_diag, inv_sqrt_diag)
    return W, [], []



def partition_localgeometry(newpoints, method, removemean):
    """
    Computes the similarity matrix using partition-based local geometry.
    
    Parameters:
        newpoints (numpy.ndarray): Query points (features × num_points × q).
        method (dict): Contains parameters, including 'partition'.
        removemean (int, optional): If 1, uses correlation; if 0, uses cosine affinity.
    
    Returns:
        scipy.sparse.csr_matrix: Similarity matrix.
    """
    points = method.get('Ref', newpoints)
    
    if points is None or points.size == 0:
        points = newpoints
    
    num_features, num_points, q = points.shape
    num_points2 = newpoints.shape[1]
    
    if newpoints.shape[0] != num_features:
        raise ValueError("Dimension mismatch between new points and reference points")
    
    sim_mat = np.zeros((num_points2, num_points))
    partition = np.array(method['partition'])
    fold_count = len(np.unique(partition))
    
    if len(partition) != num_features or min(partition) != 0 or max(partition) != fold_count-1:
        # Default to normalized inner-product if partition is not valid
        normat = np.sqrt(np.nansum(points * np.conj(points), axis=0))
        normat2 = np.sqrt(np.nansum(newpoints * np.conj(newpoints), axis=0))
        points = points / (np.finfo(float).tiny + np.tile(normat, (num_features, 1)))
        newpoints = newpoints / (np.finfo(float).tiny + np.tile(normat2, (num_features, 1)))
        points = points / np.expand_dims((np.finfo(float).tiny + np.tile(normat.T, (num_features, 1))),axis = 2)
        newpoints = newpoints / np.expand_dims((np.finfo(float).tiny + np.tile(normat2.T, (num_features, 1))),axis = 2)
        np.nan_to_num(newpoints, copy=False)
        np.nan_to_num(points, copy=False)
        sim_mat = np.abs(newpoints.T @ np.conj(points)) / q
    else:
        cI = 0  
        for fold_loop in range(0, fold_count + 1):
            I = np.where(partition == fold_loop)[0]
            
            if len(I) > 1:
                cI += len(I)
                mat = points[I, :, :]
                mat2 = newpoints[I, :, :]
                
                if removemean:
                    mat -= np.nanmean(mat, axis=0, keepdims=True)
                    mat2 -= np.nanmean(mat2, axis=0, keepdims=True)
                
                normat = np.sqrt(np.sum(mat * np.conj(mat), axis=0))
                normat2 = np.sqrt(np.sum(mat2 * np.conj(mat2), axis=0))
                
                #mat /= (np.finfo(float).eps + normat)
                mat = mat / np.expand_dims((np.finfo(float).tiny + np.tile(normat.T, (len(I), 1))),axis=2)
                #mat2 /= (np.finfo(float).eps + normat2)
                mat2 = mat2 / np.expand_dims((np.finfo(float).tiny + np.tile(normat2.T, (len(I), 1))),axis=2)
                mat = mat.transpose(0, 2, 1).reshape((len(I) * q, num_points))
                mat2 = mat2.transpose(0, 2, 1).reshape((len(I) * q, num_points2))
                
                mat_result = np.abs(mat2.T @ np.conj(mat))
                sim_mat += mat_result * len(I)
                
        if cI > 0:
            sim_mat /= (q * cI)
    
    return sim_mat

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
    return col_aff, coords
    
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


def partition_dualgeometry_fiedler_ref(f, levels, alpha, removemean, min_elements = 1):
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
            _, counts = np.unique(levels.level_partition(i), return_counts=True)
            if np.max(counts) < min_elements:
                break
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
    UD, S, _ = svds(W_tilde, k= 2)
    U = V2 @ UD
    S = S ** 2
    idx = np.argsort(S)[::-1]  # indices of S in descending order
    S = S[idx]
    U = U[:, idx]
    U = U[:,1]
    return U, W, cols



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
    D = W @ np.asarray(W.sum(axis=0)).flatten()
    V2 = 1.0 / np.sqrt(D + 1e-12)
    V2 = csr_matrix(np.diag(V2))
    W_tilde = V2 @ W
    UD, S, _ = svds(W_tilde, k= 2)
    U = V2 @ UD
    S = S ** 2
    idx = np.argsort(S)[::-1]  # indices of S in descending order
    S = S[idx]
    U = U[:, idx]
    U = U[:,1]
    return U

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

        norm1 = np.linalg.norm(mat, axis=0) + 1e-12
        norm2 = np.linalg.norm(mat2, axis=0) + 1e-12

        mat = mat / norm1[np.newaxis, :, :]
        mat2 = mat2 / norm2[np.newaxis, :, :]

        mat = mat.transpose(0, 2, 1).reshape(len(I) * q, num_points)
        mat2 = mat2.transpose(0, 2, 1).reshape(len(I) * q, num_points2)

        mat_result = np.abs(mat2.T @ mat)
        sim_mat += mat_result * len(I)

    if cI > 0:
        sim_mat /= (q * cI)

    return sim_mat