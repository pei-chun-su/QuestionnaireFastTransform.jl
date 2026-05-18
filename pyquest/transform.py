"""
transform.py: Defines various tree transform matrices and their application to data
"""
import numpy as np
import scipy.stats


def bitree_partiton(row_tree,col_tree,row_part,col_part):
    nrows = row_tree.size
    ncols = col_tree.size
    
    row_part = np.add(row_tree.level_partition(row_part),1)
    col_part = np.add(col_tree.level_partition(col_part),1)
    
    row_part_inds = np.tile(row_part[np.newaxis].T, (1,ncols)).flatten()
    col_part_inds = np.tile(col_part, (nrows,1)).flatten()
    
    partiton_mat = np.zeros(nrows*ncols)
    k = 0
    
    for i in np.unique(row_part):
        for j in np.unique(col_part):
            inds = np.logical_and((row_part_inds == i) ,(col_part_inds == j))
            partiton_mat[inds] = k
            k = k+1
                
    return partiton_mat.reshape((nrows,ncols))


def calc_1demd_transform(data,row_tree, alpha=1.0, beta=0.0,
                         exc_sing=False, exc_raw=False):
    """
     Calculates 2D EMD transform on database of data using a tree on the rows and columns.
     each level is weighted by 2**((1-level)*alpha)
     each folder size (fraction) is raised to the beta power for weighting.
     """
    nrows,ncols = np.shape(data)
    assert nrows == row_tree.size, "Tree size must match # rows in data."
    
    row_folder_fraction = np.array([((node.size*1.0/nrows)**beta)*
                                    (2.0**((1.0-node.level)*alpha))
                                    for node in row_tree])
        
    if exc_sing:
        for node in row_tree:
            if node.size == 1:
                row_folder_fraction[node.idx] = 0.0

    coefs = averaging(data,row_tree)
    avgs = np.diag(row_folder_fraction).dot(coefs)
    
    if exc_raw:
        row_singletons_start = row_tree.tree_size - nrows
        avgs = avgs[:row_singletons_start,:]

    return avgs

def calc_2demd_transform(data,row_tree, col_tree, row_alpha=1.0, row_beta=0.0,
                         col_alpha=1.0, col_beta=0.0, exc_sing=False, exc_raw=False):
    """
     Calculates 2D EMD transform on database of data using a tree on the rows and columns.
     each level is weighted by 2**((1-level)*alpha)
     each folder size (fraction) is raised to the beta power for weighting.
     """
    nrows,ncols = np.shape(data)
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

    coefs = joint_averaging(data,row_tree,col_tree)
    avgs = folder_frac * coefs
    
    if exc_raw:
        m,n = np.shape(data)
        col_singletons_start = col_tree.tree_size - n
        row_singletons_start = row_tree.tree_size - m
        avgs = avgs[:row_singletons_start,:col_singletons_start]

    return avgs

def tree_sums_mat(row_tree,return_nodes=False):
    node_ids = np.zeros(row_tree.tree_size,int)
    node_ids[0] = -1
    mat = np.zeros((row_tree.tree_size,row_tree.size))
    for node in row_tree.traverse():
        for i in node.elements:
            mat[node.idx,i] = 1
            node_ids[node.idx] = node.idx
    
    if return_nodes:
        return mat, node_ids
    else:
        return mat

def tree_averages_mat(row_tree,return_nodes=False):
    mat, node_ids = tree_sums_mat(row_tree,return_nodes=True)
    for node in row_tree.traverse():
        mat[node.idx,:] /= node.size
    if return_nodes:
        return mat, node_ids
    else:
        return mat

def tree_differences_mat(row_tree,return_nodes=False):
    node_ids = np.zeros(row_tree.tree_size,int)
    node_ids[0] = -1
    mat_avg = tree_averages_mat(row_tree)
    mat = np.zeros(np.shape(mat_avg))
    for node in row_tree.traverse():
        if node.parent is None:
            mat[node.idx,:] = mat_avg[node.idx,:]
        else:
            mat[node.idx,:] = mat_avg[node.idx,:] - mat_avg[node.parent.idx,:]
        node_ids[node.idx] = node.idx
    
    if return_nodes:
        return mat, node_ids
    else:
        return mat

def entropy(data,row_tree,col_tree):
    coefs = joint_difference(data,row_tree,col_tree)
    return np.sum(np.absolute(coefs))

def averaging(data,tree):
    nrows,ncols = np.shape(data)
    if nrows == tree.size:
        row_avg_mat = tree_averages_mat(tree)
        coefs = row_avg_mat.dot(data)
    elif ncols == tree.size:
        col_avg_mat = tree_averages_mat(tree)
        coefs = data.dot(col_avg_mat.T)
    return coefs

def difference(data,tree):
    nrows,ncols = np.shape(data)
    if nrows == tree.size:
        row_diff_mat = tree_differences_mat(tree)
        coefs = row_diff_mat.dot(data)
    elif ncols == tree.size:
        col_diff_mat = tree_differences_mat(tree)
        coefs = data.dot(col_diff_mat.T)
    return coefs

def joint_averaging(data,row_tree,col_tree):
    nrows,ncols = np.shape(data)
    assert nrows == row_tree.size, "Tree size must match # rows in data."
    assert ncols == col_tree.size, "Tree size must match # cols in data."

    row_avg_mat = tree_averages_mat(row_tree)
    coefs = row_avg_mat.dot(data)
    col_avg_mat = tree_averages_mat(col_tree)
    coefs = coefs.dot(col_avg_mat.T)
    return coefs

def joint_difference(data,row_tree,col_tree):
    nrows,ncols = np.shape(data)
    assert nrows == row_tree.size, "Tree size must match # rows in data."
    assert ncols == col_tree.size, "Tree size must match # cols in data."
    
    row_diff_mat = tree_differences_mat(row_tree)
    coefs = row_diff_mat.dot(data)
    col_diff_mat = tree_differences_mat(col_tree)
    coefs = coefs.dot(col_diff_mat.T)
    return coefs
