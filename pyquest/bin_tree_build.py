"""
bin_tree_build.py: Functions for building trees based on cuts of the first
                   nontrivial eigenvector of a diffusion.
"""

import Mytree as tree
import markov
import numpy as np
import dual_affinity
from joblib import Parallel, delayed

def build_simple_bisect_tree(node):
    """
    Recursively splits a node's indices into two equal (or near-equal) halves.
    """
    n = node.size
    
    if n <= 1:
        return
    
    # Calculate the split point
    mid = n // 2
    
    # Create a mask or index array to define the cut
    # Indices 0 to mid-1 go to the left child (0)
    # Indices mid to n-1 go to the right child (1)
    cut = np.zeros(n, dtype=int)
    cut[mid:] = 1
    
    # Apply the cut to create children
    node.create_subclusters(cut)
    
    # Recursively build the subtrees
    for child in node.children:
        build_simple_bisect_tree(child)

def bin_tree_build(affinity,cut_type="r_dyadic",bal_constant=1.0,diag_bias = False):
    """
    Takes a static, square, symmetric nxn affinity on n nodes and 
    applies the second eigenvector binary cut algorithm to it.
    cut_types currently supported are: 
    r_dyadic:   random dyadic; uniform distribution on the legal splits
                based on the balance constant.
    zero:       splits the eigenvector at zero, subject to the balance constant 
    """
    
    _,n = affinity.shape

    root = tree.ClusterTreeNode(range(n))
    queue = [root]

    while max([x.size for x in queue]) > 1:
        new_queue = []
        for node in queue:
            if node.size > 2:
                #cut it
                if cut_type == "zero":
                    cut = zero_eigen_cut(node,affinity,diag_bias)
                elif cut_type == "r_dyadic":
                    left,right = bal_cut(node.size,bal_constant)
                    cut = random_dyadic_cut(node, affinity, left, right)
                node.create_subclusters(cut)
            else:
                #make the singletons
                node.create_subclusters(np.arange(node.size))
            new_queue.extend(node.children)
        queue = new_queue

    root.make_index()                
    return root    

def zero_eigen_cut(node,affinity,diag_bias):
    """
    Returns the cut of the affinity matrix (cutting at zero) 
    corresponding to the elements in node, under the condition of bal_constant.
    """ 
    new_data = affinity[node.elements,:][:,node.elements]
    
    vecs,_ = markov.markov_eigs(new_data, 2,diag_bias)
    labels = vecs[:,1] < 0.0
    
    return labels

def random_dyadic_cut(node,affinity,left,right,diag_bias = False):
    """
    Returns a randomized cut of the affinity matrix (cutting at zero) 
    corresponding to the elements in node, under the condition of bal_constant.
    """ 
    new_data = affinity[node.elements,:][:,node.elements]
    
    vecs,_ = markov.markov_eigs(new_data, 2,diag_bias)
    eig = vecs[:,1]
    eig_sorted = eig.argsort().argsort()
    cut_loc = np.random.randint(left,right+1)
    labels = eig_sorted < cut_loc
    
    return labels
    
def bal_cut(n,balance_constant):
    """
    Given n nodes and a balance_constant, returns the endpoints of the 
    interval of legal cutpoints for a binary tree.
    """ 
    if n==1:
        return 0,1
    left = int(np.ceil((1.0/(1.0+balance_constant))*n))
    right = int(np.floor((balance_constant/(1.0+balance_constant))*n))
    if left > right and n % 2 == 1:
        left = int(np.floor(n/2.0))
        right = int(np.ceil(n/2.0))
    elif left > right:
        left = right
    return left,right    

def random_binary_tree(n,bal_constant):
    """
    Creates a random binary tree on n nodes that conforms to the balance
    constant.
    """
    root = tree.ClusterTreeNode(range(n))
    queue = [root]
    while queue:
        if (max([x.size for x in queue]) == 1 and 
            max([x.level for x in queue]) == min([x.level for x in queue])):
            break
        node = queue.pop(0)
        left,right = bal_cut(node.size, bal_constant)
        cut_loc = np.random.randint(left,right+1)
        labels = np.array(node.elements).argsort().argsort() < cut_loc
        node.create_subclusters(labels)
        queue.extend(node.children)
    root.make_index()
    return root

def bin_tree_build4(total_affinity,cut_type="r_dyadic",bal_constant=1.0,diag_bias = False):
    """
    Takes a static, square, symmetric nxn affinity on n nodes and 
    applies the second eigenvector binary cut algorithm to it.
    cut_types currently supported are: 
    r_dyadic:   random dyadic; uniform distribution on the legal splits
                based on the balance constant.
    zero:       splits the eigenvector at zero, subject to the balance constant 
    """
    _,n = total_affinity.shape
    root = tree.ClusterTreeNode(range(n))
    queue = [root]
    node = queue[0]
    new_queue = []
    G1_c = []
    current_affinity = total_affinity[np.ix_(node.elements, node.elements)]
    vecs,_ = markov.markov_eigs(current_affinity, 2,diag_bias)
    eig = vecs[:,1]
    left,right = bal_cut(node.size,bal_constant)
    eig_sorted = eig.argsort().argsort()
    cut_loc = np.random.randint(left,right+1)
    labels = eig_sorted >= cut_loc
    node.create_subclusters(labels)
    elements_array = np.array(node.elements)
    g0 = elements_array[labels == False]
    g1 = elements_array[labels == True]
    if np.median(eig[g0]) > np.median(eig[g1]):
        eig = -eig
    G0 = g0[np.argsort(eig[g0])[-1]]
    G1 = g1[np.argsort(eig[g1])[-1]]
    G1_c.append(G0)
    G1_c.append(G1)
    new_queue.extend(node.children)
    queue = new_queue
    rs_c = []
    rs_c.append([1,n+1])
    rs_c.append([1,len(g1)+1,n+1])
    
    while max([x.size for x in queue]) > 1:
        new_queue = []
        G1_c_new = []
        l = 0
        rs = []
        for node in queue:
            if node.size > 2:
                current_affinity = total_affinity[np.ix_(node.elements, node.elements)]
                vecs,_ = markov.markov_eigs(current_affinity, 2,diag_bias)
                eig = vecs[:,1]
                left,right = bal_cut(node.size,bal_constant)
                eig_sorted = eig.argsort().argsort()
                cut_loc = np.random.randint(left,right+1)
                labels = eig_sorted >= cut_loc          
                elements_array = np.array(node.elements)
                g0 = elements_array[labels == False]
                g1 = elements_array[labels == True]
                connect = G1_c[l]
                if connect not in g1:
                        labels = ~ labels
                        eig = - eig
                g0 = elements_array[labels == False]
                g1 = elements_array[labels == True]
                G0 = g0[np.argsort(eig[labels == False])[-1]]
                G1_c_new.append(int(G0))
                G1_c_new.append(int(connect))
                node.create_subclusters(labels)
                rs.extend([len(g0),len(g1)])
                
            else:
                #make the singletons
                if (len(node.elements)  == 1):
                    connect = G1_c[l]
                    G1_c_new.append(connect)
                    node.create_subclusters(np.arange(node.size))
                    rs.extend([1])
                else:
                    connect = G1_c[l]
                    elements_array = np.array(node.elements)
                    labels = (elements_array == connect)
                    G0 = elements_array[labels == False]
                    G1_c_new.append(int(G0[0]))
                    G1_c_new.append(int(connect))
                    node.create_subclusters(~labels)
                    rs.extend([1,1])
            l+=1
            new_queue.extend(node.children)
        G1_c = G1_c_new
        queue = new_queue
        rs_c.append(rs[::-1])
    
    root.make_index()
    for i in range(2,len(rs_c)):
        rs_c[i] = np.insert(np.cumsum(rs_c[i]) + 1, 0, 1).tolist()

    return root,G1_c[::-1], rs_c