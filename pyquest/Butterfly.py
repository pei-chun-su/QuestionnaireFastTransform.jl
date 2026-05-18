import Mytree
import numpy as np
from scipy import linalg
from scipy.linalg import qr, svdvals, solve_triangular, norm

def dyadic_partition(data,row_tree,col_tree, C_max = 128):
    for i in range(1,col_tree.tree_depth+1):
        queue = col_tree.dfs_level(i)
        if max([x.size for x in queue])> C_max:
            continue
        else:
            level = i
            break
    level_blocks = {} 
    level_infos = {}

    for i in range(level):
        # Initialize the specific list for this level
        
        queue_col = col_tree.dfs_level(level-i)
        queue_row = row_tree.dfs_level(i+1)
        for k in range(len(queue_col)):
            for j in range(len(queue_row)):
                row_elements = np.array(queue_row[j].elements)
                col_elements = np.array(queue_col[k].elements)
                
                block_content = {
                    "level": i,
                    "j": j,
                    "k": k,
                    "block": data[np.ix_(row_elements, col_elements)],
                    "row_range": row_elements, 
                    "col_range": col_elements
                }
                # Append to the list FOR THIS LEVEL ONLY
                level_blocks[(i,j,k)] = block_content
                
                info_content = {
                    "level": i,
                    "j": j, 
                    "k": k, 
                    "row_range": row_elements, 
                    "col_range": col_elements
                }
                level_infos[(i,j,k)] = info_content
        
                
    return level_blocks, level_infos


import numpy as np
from scipy.linalg import qr, svdvals, norm

def id_col(A, acc,rank = None):
    normA = norm(A)
    m, n = A.shape
    dim = min(m, n)
    
    if dim <= 2000:
        ss = svdvals(A)
        s = 1
        k = max(np.sum(ss / ss[0] > 0.1 * acc * normA), s)
        
        Q, R, ind = qr(A, pivoting=True, mode='economic')
        
        T = np.linalg.solve(R[:k, :k], R[:k, k:])
        Z = np.zeros((k, n))
        Z[:, ind] = np.hstack([np.eye(k), T])
        Js = ind[:k]
        
        return Js, Z, k
    else:
        print("randomized method is used.")
        Q, B, ss = random_qb(A, acc)
        s = 1
        k = max(np.sum(ss / ss[0] > 0.1 * acc * normA), s)
        
        _, R, ind = qr(B, pivoting=True, mode='economic')
        
        T = np.linalg.solve(R[:k, :k], R[:k, k:])
        Z = np.zeros((k, n))
        Z[:, ind] = np.hstack([np.eye(k), T])
        Js = ind[:k]
        
        return Js, Z, k

def random_qb(A, acc):
    normA = norm(A)
    m, n = A.shape
    s = min(m, n)
    b = min(450, s)
    tol = 0.1 * acc * normA * np.sqrt(s)
    
    Q = np.zeros((m, 0))
    B = np.zeros((0, n))
    
    for j in range(1, s + 1):
        G = np.random.randn(n, b)
        U, _ = qr(A @ G, mode='economic')
        
        if j > 1:
            U, _ = qr(U - Q @ (Q.T @ U), mode='economic')
        
        newB = U.T @ A
        B = np.vstack([B, newB])
        Q = np.hstack([Q, U])
        A = A - U @ newB
        
        if norm(A) < tol:
            ss = svdvals(B)
            k = int(np.round(0.97 * len(ss)))
            if ss[k] > 0.1 * normA * acc:
                tol /= 2
            else:
                return Q, B, ss
    
    ss = svdvals(B)
    return Q, B, ss


"""
def random_qb(A, acc):
    # We work on a copy to avoid destroying the original matrix A
    R = A.copy() 
    m, n = R.shape
    norm_A = linalg.norm(A)
    s = min(m, n)
    b = min(450, s)
    tol = 0.1 * acc * norm_A * np.sqrt(s)

    Q_total = np.zeros((m, 0))
    B_total = np.zeros((0, n))

    for j in range(s):
        # 1. Sketching step
        G = np.random.standard_normal((n, b))
        Y = R @ G
        
        # 2. Orthonormalize current block
        U, _ = linalg.qr(Y, mode='economic')
        
        # 3. Re-orthogonalize against existing Q
        if Q_total.shape[1] > 0:
            # U = U - Q(Q'U)
            U = U - Q_total @ (Q_total.T @ U)
            # Second pass for numerical stability
            U, _ = linalg.qr(U, mode='economic')
        
        # 4. Project and update
        new_B = U.T @ R
        B_total = np.vstack([B_total, new_B])
        Q_total = np.hstack([Q_total, U])
        
        # 5. Deflate the residual
        R = R - U @ new_B
        
        # 6. Convergence Check
        curr_norm = linalg.norm(R)
        if curr_norm < tol:
            ss = linalg.svdvals(B_total)
            # Your specific 97% logic
            k_idx = int(0.97 * len(ss)) - 1
            if ss[k_idx] > 0.1 * norm_A * acc:
                tol /= 2
            else:
                return Q_total, B_total, ss
                
    ss = linalg.svdvals(B_total)
    return Q_total, B_total, ss



def id_col(A, acc=1e-12,rank = None):
    m, n = A.shape
    norm_A = norm(A)

    if n <= 2**10:
        # Standard deterministic path
        _, R, p = qr(A, pivoting=True)
    else:
        l = min(n, 1024) 
        Omega = np.random.standard_normal((n, l))
        Y = A @ Omega
        
        Q_basis, _ = qr(Y, mode='economic')
        B = Q_basis.T @ A
        _, R, p = qr(B, pivoting=True)

    # --- Adaptive Rank Determination ---
    diag_R = np.abs(np.diag(R))
    # Find k where the diagonal of R drops below our threshold
    # Julia's ss[1] equivalent is diag_R[0]
    k_acc = np.searchsorted(-diag_R, -0.1 * acc * norm_A) if diag_R[0] > 0 else 0
    if rank is None:
        k = max(int(k_acc), 1)
    else:
        k = int(np.clip(k_acc, 1, rank))
    # --- ID Solver ---
    Js = p[:k]
    R11 = R[:k, :k] +1e-12
    R12 = R[:k, k:]
    
    T = solve_triangular(R11, R12, check_finite=False)
    
    Z_pivoted = np.hstack([np.eye(k), T])
    Z = np.zeros((k, n))
    Z[:, p] = Z_pivoted
    
    return Js, Z, k
"""
def compress_dyadic_blocks(level_blocks,acc = 1e-10,rank = None):
    """
    Python version of the Butterfly compression.
    level_blocks: dictionary keyed by level i (0 to L)
    acc: accuracy tolerance for ID_col
    """
    # L is the maximum level index (e.g., if keys are 0, 1, 2, then L = 2)
    L = max(k[0] for k in level_blocks.keys())
    
    B = {}   # Skeletons (B-factors)
    BL = {}  # Final level skeletons
    P = {}   # Interpolation matrices (P-factors)

    # === Step 1: Level 0 (Initial Base Blocks) ===
    # Here, columns are at their finest level, rows are at their coarsest
    INFO = [item for item in level_blocks if item[0] == 0]
    for block_info in INFO:
        j = block_info[1]
        k = block_info[2]
        A = level_blocks[block_info]['block']
        # ID_col: A ≈ A[:, Js] @ Z
        Js, P0, r = id_col(A, acc,rank)
        U = A[:, Js]
        B[(0, j, k)] = U
        P[(0, j, k)] = P0
     
    print('Level 0 Done')

    # === Step 2: Hierarchical compression for levels i = 1 to L ===
    for i in range(1, L + 1):
        INFO = [item for item in level_blocks if item[0] == i]
        for block_info in INFO:
            j = block_info[1]
            k = block_info[2]
            # Parent index in row direction (j0)
            # Python 0-indexed version of fld(j+1, 2) is simply j // 2
            j0 = j // 2
            # Get skeletons from previous level (i-1)
            # Butterfly logic: parent k is formed by merging children 2k and 2k+1
            A1 = B[(i - 1, j0, 2 * k)]
            A2 = B[(i - 1, j0, 2 * k + 1)]
            row_range_pre = level_blocks[(i - 1, j0, 2 * k)]['row_range']
            row_range_cur =  level_blocks[block_info]['row_range']
            ind = np.searchsorted(row_range_pre,row_range_cur)
            

            # Determine top or bottom halves based on row splitting
           
            A1_half = A1[ind, :]
            A2_half = A2[ind, :]
           
            # Horizontal concatenation of child skeletons
            A_merged = np.hstack([A1_half, A2_half])

            # Interpolative Decomposition on merged skeletons
            Js, Pk, r = id_col(A_merged, acc,rank)
            U = A_merged[:, Js]

            # If we are at the top level, store in the special BL dictionary
            if i == L:
                BL[(i, j)] = U
                B[(i, j)] = U # for consistency
                P[(i, j)] = Pk
            else:
                B[(i, j, k)] = U
                P[(i, j, k)] = Pk
        print(f"Level {i} Done")

    return BL, P

def apply_compressed_operator(BL, P, alpha, level_infos):
    """
    Python version of the Butterfly Matrix-Vector Product.
    B, P: Dictionaries containing the compressed factors.
    alpha: The input vector to be multiplied.
    level_infos: Dictionary keyed by level containing 'col_range' and 'row_range'.
    """
    # L is the maximum level index (e.g., if keys are 0, 1, 2, L = 2)
    L = max(key[0] for key in P.keys())
    beta = {}

    # --- Step 1: Apply first-level interpolation (Level 0) ---
    # In level_infos[0], row blocks are largest, col blocks are finest.
    INFO = [item for item in level_infos if item[0] == 0]
    for info in INFO:
        j = info[1]
        k = info[2]
        col_indices = level_infos[info]['col_range']
        alpha_block = alpha[col_indices]
        
        # beta[(0, j, k)] = P[(0, j, k)] * alpha_block
        beta[(0, j, k)] = P[(0, j, k)] @ alpha_block

    # --- Step 2: Recursive interpolation from i = 1 to L ---
    # Note: Your Julia code had a redundant loop for Step 2; 
    # This single loop handles everything up to the final level L.
    for i in range(1, L + 1):
        # We iterate based on keys existing in B or P for this level
        INFO = [item for item in level_infos if item[0] == i]
        for info in INFO:
            j = info[1]
            k = info[2]
            j_parent = j // 2  # Parent row index in 0-based indexing

            # Get beta vectors from the previous (finer column) level
            # Children of k are 2k and 2k+1
            b1 = beta[(i - 1, j_parent, 2 * k)]
            b2 = beta[(i - 1, j_parent, 2 * k + 1)]
            
            # vcat equivalent
            beta_combined = np.concatenate([b1, b2])

            # Determine the key (last level uses (L, j), others use (i, j, k))
            if i == L:
                key = (i, j)
            else:
                key = (i, j, k)
            
            beta[key] = P[key] @ beta_combined

    # --- Step 3: Final matrix application at level L ---
    INFO = [item for item in level_infos if item[0] == L]
    level_L_js = {key[1] for key in level_infos.keys() if key[0] == L}
    num_j_blocks= len(level_L_js)
    row_order = []
    f_pre = []
    for j in range(num_j_blocks):
        beta_L_j = beta[(L, j)]
        B_L_j = BL[(L, j)]

        assert B_L_j.shape[1] == len(beta_L_j), f"Mismatch at j={j}"
        
        # Calculate resulting segment
        f_pre.append(B_L_j @ beta_L_j)
        row_range =  level_infos[(L,j,0)]['row_range']
        row_order.append(row_range)
    # --- Step 4: Concatenate all output blocks ---
    # Sort by j to ensure the vector is reconstructed in the correct row order
    row_order = np.concatenate(row_order)
    f_pre = np.concatenate(f_pre)
    f_final = np.zeros(len(row_order))
    f_final[row_order] = f_pre
    
    return f_final
    
