import numpy as np

class GraphPart:
    """Class to hold the GHWT results and parameters."""
    def __init__(self, rs):
        self.rs = rs
        self.tag = None
        self.compinfo = None
        self.dmatrix = None
        self.ind = None

def ghwt_core(rs_list, order):
    # 1. Preliminaries
    jmax = len(rs_list)
    N = rs_list[0][-1] - 1 
    
    tag = np.zeros((N, jmax), dtype=int)
    compinfo = np.zeros((N, jmax), dtype=int)

    # 2. Perform the transform logic
    for j in range(jmax - 2, -1, -1):
        regioncount = len(rs_list[j]) - 1
        for r in range(regioncount):
            rs1 = rs_list[j][r] - 1
            rs3 = rs_list[j][r + 1] - 1
            n = rs3 - rs1
            
            if n > 1:
                rs2 = rs1 + 1
                while rs2 < rs3 and tag[rs2, j + 1] != 0:
                    rs2 += 1
                    
                if rs2 == rs3:
                    tag[rs1 : rs3, j] = tag[rs1 : rs3, j + 1]
                else:
                    n1 = rs2 - rs1
                    n2 = rs3 - rs2
                    compinfo[rs1, j] = n1
                    compinfo[rs1 + 1, j] = n2
                    tag[rs1 + 1, j] = 1
                    
                    parent = rs1 + 2
                    child1 = rs1 + 1
                    child2 = rs2 + 1
                    
                    while parent < rs3:
                        if child1 < rs2 and (child2 == rs3 or tag[child1, j + 1] < tag[child2, j + 1]):
                            tag[parent, j] = 2 * tag[child1, j + 1]
                            child1 += 1
                            parent += 1
                        elif child2 < rs3 and (child1 == rs2 or tag[child2, j + 1] < tag[child1, j + 1]):
                            tag[parent, j] = 2 * tag[child2, j + 1]
                            child2 += 1
                            parent += 1
                        else:
                            tag[parent, j] = 2 * tag[child1, j + 1]
                            tag[parent + 1, j] = tag[parent, j] + 1
                            compinfo[parent, j] = 1
                            compinfo[parent + 1, j] = 1
                            child1 += 1
                            child2 += 1
                            parent += 2
            elif n <= 0:
                raise ValueError(f"n must be positive: n = {n}")

    # Create the object and assign attributes
    GP = GraphPart(rs_list)
    GP.tag = tag
    GP.compinfo = compinfo
    GP.ind = order
    return GP


def ghwt_core2(rs, dmatrix, order):
    """
    Python implementation of GHWT Core with Scaling, Haar, and Walsh basis.
    
    Parameters:
    rs: List of lists containing region boundaries (1-based from Julia).
    dmatrix: 3D NumPy array (N, jmax, num_signals).
    
    Returns:
    dmatrix, tag, compinfo
    """
    jmax = len(rs)
    N = rs[0][-1] - 1  # Total length based on Level 0
    num_signals = dmatrix.shape[2]

    # Initialize matrices
    tag = np.zeros((N, jmax), dtype=int)
    compinfo = np.zeros((N, jmax), dtype=int)

    # 1. Perform the transform
    # From jmax-2 down to 0 (Level j inherits from Level j+1)
    for j in range(jmax - 2, -1, -1):
        regioncount = len(rs[j]) - 1
        
        for r in range(regioncount):
            # Convert 1-based Julia boundaries to 0-based Python indices
            rs1 = rs[j][r] - 1
            rs3 = rs[j][r + 1] - 1
            n = rs3 - rs1
            
            # Case 1: Single node region
            if n == 1:
                dmatrix[rs1, j, :] = dmatrix[rs1, j + 1, :]
            
            # Case 2: Multi-node region
            elif n > 1:
                rs2 = rs1 + 1
                # Find the split point in level j+1
                while rs2 < rs3 and tag[rs2, j + 1] != 0:
                    rs2 += 1
                
                # If no split point found: Parent is a copy
                if rs2 == rs3:
                    dmatrix[rs1:rs3, j, :] = dmatrix[rs1:rs3, j + 1, :]
                    tag[rs1:rs3, j] = tag[rs1:rs3, j + 1]
                
                # Parent region has 2 child regions: Perform Math
                else:
                    n1 = rs2 - rs1
                    n2 = rs3 - rs2
                    sqrt_n = np.sqrt(n)
                    sqrt_n1 = np.sqrt(n1)
                    sqrt_n2 = np.sqrt(n2)

                    # --- SCALING COEFFICIENT ---
                    dmatrix[rs1, j, :] = (
                        sqrt_n1 * dmatrix[rs1, j + 1, :] +
                        sqrt_n2 * dmatrix[rs2, j + 1, :]
                    ) / sqrt_n
                    compinfo[rs1, j] = n1

                    # --- HAAR COEFFICIENT ---
                    dmatrix[rs1 + 1, j, :] = (
                        sqrt_n2 * dmatrix[rs1, j + 1, :] -
                        sqrt_n1 * dmatrix[rs2, j + 1, :]
                    ) / sqrt_n
                    compinfo[rs1 + 1, j] = n2
                    tag[rs1 + 1, j] = 1

                    # --- WALSH COEFFICIENTS ---
                    parent = rs1 + 2
                    child1 = rs1 + 1
                    child2 = rs2 + 1
                    sqrt2 = np.sqrt(2)

                    while parent < rs3:
                        # Logic to select which child tag is smaller
                        if child1 < rs2 and (child2 == rs3 or tag[child1, j + 1] < tag[child2, j + 1]):
                            dmatrix[parent, j, :] = dmatrix[child1, j + 1, :]
                            tag[parent, j] = 2 * tag[child1, j + 1]
                            child1 += 1
                            parent += 1
                        
                        elif child2 < rs3 and (child1 == rs2 or tag[child2, j + 1] < tag[child1, j + 1]):
                            dmatrix[parent, j, :] = dmatrix[child2, j + 1, :]
                            tag[parent, j] = 2 * tag[child2, j + 1]
                            child2 += 1
                            parent += 1
                            
                        else:
                            # Matching coefficients: Combine using sum/difference
                            # tag[parent, j] is the 'average' of frequencies
                            tag[parent, j] = 2 * tag[child1, j + 1]
                            dmatrix[parent, j, :] = (
                                dmatrix[child1, j + 1, :] + 
                                dmatrix[child2, j + 1, :]
                            ) / sqrt2
                            compinfo[parent, j] = 1

                            # tag[parent + 1, j] is the 'difference' of frequencies
                            tag[parent + 1, j] = tag[parent, j] + 1
                            dmatrix[parent + 1, j, :] = (
                                dmatrix[child1, j + 1, :] - 
                                dmatrix[child2, j + 1, :]
                            ) / sqrt2
                            compinfo[parent + 1, j] = 1
                            
                            child1 += 1
                            child2 += 1
                            parent += 2
            
            else:
                raise ValueError(f"n must be positive: n = {n}")
    GP = GraphPart(rs)
    GP.tag = tag
    GP.compinfo = compinfo
    GP.ind = order
    return dmatrix,GP

def ghwt_analysis(f, rs, order):
    """
    1D GHWT Analysis.
    
    Parameters:
    f: The signal matrix (N x num_signals).
    GP:  The GraphPart object (must have .ind and .rs).
    c2f: Boolean, if True returns the full dmatrix dictionary.
    """
    # 0. Preliminaries
    ind = order
    # N is the signal length, jmax is number of levels
    jmax = len(rs)
    N = rs[0][-1] - 1
    
    # Number of signals (columns in f)
    if f.ndim == 1:
        f = f.reshape(-1, 1)
    fcols = f.shape[1]

    # Allocate space for expansion coefficients (Dictionary Matrix)
    # Shape: (Space, Levels, Signals)
    dmatrix = np.zeros((N, jmax, fcols))

    # Permute the signal into the order dictated by the partition tree
    # Julia: dmatrix[:, jmax, :] = f[ind, :]
    # Python: We put it in the last level (jmax - 1)
    dmatrix[:, jmax - 1, :] = f[ind, :]

    # 1. Perform the transform
    # This updates tag, compinfo, and fills the dmatrix levels
    # We use the verified ghwt_core2 function from our previous discussion
    dmatrix, GP = ghwt_core2(rs, dmatrix, order)
    
    # Store results back into the GP object for persistence
  
    return dmatrix, GP

def ghwt_analysis_2d(matrix, rs_rows, rs_cols, row_order,col_order):
    matrix_re = matrix[row_order][:,col_order]
    frows, fcols = matrix_re.shape
    
    # --- STEP 1: ROW EXPANSION ---
    jmax_row = len(rs_rows)
    N_row = rs_rows[0][-1] - 1
    dmatrix_row = np.zeros((N_row, jmax_row, fcols))
    dmatrix_row[:, jmax_row - 1, :] = matrix_re
    
    # Call your 1D core
    dmatrix_row, GProws = ghwt_core2(rs_rows, dmatrix_row, row_order)
    temp_reshaped = dmatrix_row.reshape(N_row * jmax_row, fcols, order='F')
    dmatrix_transposed = temp_reshaped.T  # Shape: (fcols, N_row * jmax_row)

    # --- STEP 2: COLUMN EXPANSION ---
    jmax_col = len(rs_cols)
    N_col = rs_cols[0][-1] - 1
    num_signals_new = dmatrix_transposed.shape[1]
    
    dmatrix_col = np.zeros((N_col, jmax_col, num_signals_new))
    dmatrix_col[:, jmax_col - 1, :] = dmatrix_transposed
    
    dmatrix_col, GPcols = ghwt_core2(rs_cols, dmatrix_col,col_order)
    final_output = dmatrix_col.reshape(N_col * jmax_col, N_row * jmax_row, order='F').T
    
    return final_output, GProws, GPcols


def rs_to_region(GP):
    """
    Calculates the region tags (tag_r) based on the partition tree rs and frequency tags.
    GP: A GraphPart object containing GP.rs and GP.tag.
    Returns: A 2D NumPy array (tag_r).
    """
    rs = GP.rs
    tag = GP.tag
    N, jmax = tag.shape
    
    # Initialize the region tag matrix
    tag_r = np.zeros((N, jmax), dtype=int)
    
    # j goes from 0 up to jmax-2 (since we update j+1)
    for j in range(jmax - 1):
        # Handle the ragged rs list
        regioncount = len(rs[j]) - 1
        
        for r in range(regioncount):
            # Convert 1-based Julia boundaries to 0-based Python indices
            rs1 = rs[j][r] - 1
            rs3 = rs[j][r + 1] - 1
            s = rs3 - rs1
            
            if s == 1:
                # The region and subregion have only one element
                tag_r[rs1, j + 1] = 2 * tag_r[rs1, j]
                
            elif s > 1:
                # rs2 marks the start of the second subregion
                rs2 = rs1 + 1
                while rs2 < rs3 and tag[rs2, j + 1] != 0:
                    rs2 += 1
                
                # Check if the parent region is a copy of the subregion
                if rs2 == rs3:
                    tag_r[rs1 : rs3, j + 1] = 2 * tag_r[rs1 : rs3, j]
                
                # The parent region has 2 child regions
                else:
                    # First subregion gets 2*k
                    tag_r[rs1 : rs2, j + 1] = 2 * tag_r[rs1 : rs2, j]
                    # Second subregion gets 2*k + 1
                    tag_r[rs2 : rs3, j + 1] = 2 * tag_r[rs2 : rs3, j] + 1
                    
    return tag_r

def tf2d_init(tag, tag_r):
    """
    Creates bidirectional mapping between linear indices and (level, region, frequency) tags.
    
    Parameters:
    tag: 2D NumPy array (Frequency tags)
    tag_r: 2D NumPy array (Region tags)
    
    Returns:
    tag2ind: Dictionary mapping (j, r, f) -> linear_index
    ind2tag: Dictionary mapping linear_index -> (j, r, f)
    """
    m, n = tag.shape
    tag2ind = {}
    ind2tag = {}
    
    for j in range(n):
        for i in range(m):
            # Define the triplet: (Level, Region, Frequency)
            # Keeping j, tag_r, and tag values as they are, 
            # but calculating a 0-based linear index.
            triplet = (j, tag_r[i, j], tag[i, j])
            
            # Linear index logic: 
            # Julia: i + (j-1)*m (where i, j start at 1)
            # Python: i + j*m (where i, j start at 0)
            linear_index = i + (j * m)
            
            tag2ind[triplet] = linear_index
            ind2tag[linear_index] = triplet
            
    return tag2ind, ind2tag


def tf_core_2d_col(dmatrix, tag2ind, ind2tag, jmax):
    """
    Python implementation of time-frequency cost comparison for basis selection.
    
    Parameters:
    dmatrix: 2D NumPy array of coefficients.
    tag2ind: Dictionary mapping (j, r, f) -> index.
    ind2tag: Dictionary mapping index -> (j, r, f).
    jmax: Maximum number of levels.
    """
    # Power of the cost-functional (p=1 corresponds to l1 norm/sparsity)
    costp = 1
    p, q = dmatrix.shape
    
    # Initialize output structures
    # We use a list for new dictionaries and prune at the end
    dmatrix_new = np.zeros((p, q))
    tag_tf = np.zeros((p, q), dtype=bool)
    tag2ind_new = {}
    ind2tag_new = {}

    s = 0  # Python is 0-indexed
    
    # Iterate through columns (q is the number of dictionary atoms)
    for i in range(q):
        # Find level j, region k, tag l information
        j, k, l = ind2tag[i]
        
        # Only look at even tags l to avoid double-processing pairs, 
        # and stay within level bounds
        if l % 2 == 0 and j < (jmax - 1):
            
            # 1. CALCULATE FREQUENCY COST
            # (j, k, l) and (j, k, l+1)
            pair_idx = tag2ind.get((j, k, l + 1))
            if pair_idx is not None:
                freqcos = np.abs(dmatrix[:, i])**costp + np.abs(dmatrix[:, pair_idx])**costp
            else:
                freqcos = np.abs(dmatrix[:, i])**costp
            
            # 2. CALCULATE TIME/SPATIAL COST
            # Child nodes in the tree: (j+1, 2k, l/2) and (j+1, 2k+1, l/2)
            child1_key = (j + 1, 2 * k, l // 2)
            child2_key = (j + 1, 2 * k + 1, l // 2)
            
            child1_idx = tag2ind.get(child1_key)
            child2_idx = tag2ind.get(child2_key)
            
            if child1_idx is None and child2_idx is not None:
                timecos = np.abs(dmatrix[:, child2_idx])**costp
            elif child2_idx is None and child1_idx is not None:
                timecos = np.abs(dmatrix[:, child1_idx])**costp
            elif child1_idx is not None and child2_idx is not None:
                timecos = np.abs(dmatrix[:, child1_idx])**costp + np.abs(dmatrix[:, child2_idx])**costp
            else:
                # Degenerate case: neither child exists
                timecos = np.zeros(p)

            # 3. COMPARE AND DECIDE
            # tag_tf stores the decision: 1 (True) if frequency is better/cost is higher? 
            # Note Julia: timecos >= freqcos
            decision = (timecos >= freqcos)
            tag_tf[:, s] = decision
            
            # Store the minimized cost in dmatrix_new
            dmatrix_new[:, s] = decision * freqcos + (~decision) * timecos
            
            # Update the new index mapping (using l//2 to move up frequency scale)
            new_triplet = (j, k, l // 2)
            ind2tag_new[s] = new_triplet
            tag2ind_new[new_triplet] = s
            
            s += 1

    # Prune matrices to the actual number of elements found
    dmatrix_new = dmatrix_row = dmatrix_new[:, :s]
    tag_tf = tag_tf[:, :s]
    
    return dmatrix_new, tag2ind_new, ind2tag_new, tag_tf.astype(np.uint8)


def tf_core_2d_row(dmatrix, tag2ind, ind2tag, jmax):
    """
    Performs the row-wise cost comparison for 2D Best Basis selection.
    
    Parameters:
    dmatrix: 2D NumPy array of coefficients.
    tag2ind: Dict mapping (j, r, f) -> index.
    ind2tag: Dict mapping index -> (j, r, f).
    jmax: Maximum number of levels (count).
    """
    costp = 1
    p, q = dmatrix.shape
    
    # Initialize output structures
    dmatrix_new = np.zeros((p, q))
    tag_tf = np.zeros((p, q), dtype=bool)
    tag2ind_new = {}
    ind2tag_new = {}

    s = 0  # Python 0-based index
    
    # Iterate through rows
    for i in range(p):
        # Retrieve triplet information for the current row
        # ind2tag mapping should correspond to row indices here
        j, k, l = ind2tag[i]
        
        # Process only even tags to handle pairs, and ensure we aren't at the last level
        if l % 2 == 0 and j < (jmax - 1):
            
            # 1. FREQUENCY COST (Row-wise)
            # Compare current row (l) with its pair row (l+1)
            pair_idx = tag2ind.get((j, k, l + 1))
            if pair_idx is not None:
                freqcos = np.abs(dmatrix[i, :])**costp + np.abs(dmatrix[pair_idx, :])**costp
            else:
                freqcos = np.abs(dmatrix[i, :])**costp
            
            # 2. TIME/SPATIAL COST (Row-wise)
            # Look at child levels: (j+1, 2k, l/2) and (j+1, 2k+1, l/2)
            child1_key = (j + 1, 2 * k, l // 2)
            child2_key = (j + 1, 2 * k + 1, l // 2)
            
            c1_idx = tag2ind.get(child1_key)
            c2_idx = tag2ind.get(child2_key)
            
            if c1_idx is None and c2_idx is not None:
                timecos = np.abs(dmatrix[c2_idx, :])**costp
            elif c2_idx is None and c1_idx is not None:
                timecos = np.abs(dmatrix[c1_idx, :])**costp
            elif c1_idx is not None and c2_idx is not None:
                timecos = np.abs(dmatrix[c1_idx, :])**costp + np.abs(dmatrix[c2_idx, :])**costp
            else:
                timecos = np.zeros(q)

            # 3. COMPARE AND ASSIGN
            # Julia logic: decision is True if 'time' representation is costlier/less sparse
            decision = (timecos >= freqcos)
            tag_tf[s, :] = decision
            
            # Store the minimized cost
            dmatrix_new[s, :] = decision * freqcos + (~decision) * timecos
            
            # Metadata update
            new_triplet = (j, k, l // 2)
            ind2tag_new[s] = new_triplet
            tag2ind_new[new_triplet] = s
            
            s += 1

    # Prune matrices to the actual number of rows processed
    dmatrix_new = dmatrix_new[:s, :]
    tag_tf = tag_tf[:s, :]
    
    return dmatrix_new, tag2ind_new, ind2tag_new, tag_tf.astype(np.uint8)



import gc

def ghwt_tf_bestbasis_2d(matrix, rs_rows, rs_cols, row_order, col_order):
    # 1. ANALYSIS: Generate the full redundant dictionary
    dmatrix, GProws, GPcols = ghwt_analysis_2d(matrix, rs_rows, rs_cols, row_order, col_order)
    
    jmax_row = GProws.tag.shape[1]
    jmax_col = GPcols.tag.shape[1]

    # 2. INITIALIZATION
    tag_r_row = rs_to_region(GProws)
    tag_r_col = rs_to_region(GPcols)
    
    # We maintain a "current" mapping that gets updated after every prune
    t2i_r, i2t_r = tf2d_init(GProws.tag, tag_r_row)
    t2i_c, i2t_c = tf2d_init(GPcols.tag, tag_r_col)

    # Grids for Costs and Decisions
    TAG_tf = np.empty((jmax_row, jmax_col), dtype=object)
    D_COST = np.empty((jmax_row, jmax_col), dtype=object)
    D_COST[0, 0] = np.abs(dmatrix)

    # Storage for mappings at each level (needed for Step 5 and Recovery)
    MAPS_R = [None] * jmax_row; MAPS_R[0] = (t2i_r, i2t_r)
    MAPS_C = [None] * jmax_col; MAPS_C[0] = (t2i_c, i2t_c)

    # --- STEP 3: ROW COMPARISON FIRST (Along the first column) ---
    curr_t2i, curr_i2t = MAPS_R[0]
    for i in range(jmax_row - 1):
        res_dm, next_t2i, next_i2t, res_tf = tf_core_2d_col(
            D_COST[i, 0].T, curr_t2i, curr_i2t, jmax_row - i
        )
        D_COST[i+1, 0] = res_dm.T
        TAG_tf[i+1, 0] = (res_tf.T + 2).astype(np.uint8)
        
        # UPDATE the mapping for the next row level
        curr_t2i, curr_i2t = next_t2i, next_i2t
        MAPS_R[i+1] = (curr_t2i, curr_i2t)

    # --- STEP 4 & 5: ONE COLUMN BY ANOTHER (Joint Comparison) ---
    for j in range(1, jmax_col):
        # 5a. Pure column pruning for the first row of this column
        curr_t2i_c, curr_i2t_c = MAPS_C[j-1]
        res_dm, next_t2i_c, next_i2t_c, res_tf = tf_core_2d_col(
            D_COST[0, j-1], curr_t2i_c, curr_i2t_c, jmax_col - j + 1
        )
        D_COST[0, j] = res_dm
        TAG_tf[0, j] = res_tf
        MAPS_C[j] = (next_t2i_c, next_i2t_c)

        # 5b. Joint Comparison for the rest of the grid
        for i in range(1, jmax_row):
            print((j,i))
            # Column Prune Choice (From the left)
            # Use Column Mapping from the previous col level
            t2i_c_prev, i2t_c_prev = MAPS_C[j-1]
            cost_col, _, _, tf_col = tf_core_2d_col(
                D_COST[i, j-1], t2i_c_prev, i2t_c_prev, jmax_col - j + 1
            )
            
            # Row Prune Choice (From above)
            # Use Row Mapping from the previous row level
            t2i_r_prev, i2t_r_prev = MAPS_R[i-1]
            cost_row, _, _, tf_row = tf_core_2d_col(
                D_COST[i-1, j].T, t2i_r_prev, i2t_r_prev, jmax_row - i + 1
            )
            cost_row = cost_row.T
            tf_row = (tf_row.T + 2).astype(np.uint8)

            # Joint Decision
            row_or_col = (cost_col <= cost_row)
            D_COST[i, j] = np.where(row_or_col, cost_col, cost_row)
            TAG_tf[i, j] = np.where(row_or_col, tf_col, tf_row)
        
        for k in range(jmax_row):
            D_COST[k, j-1] = None
        gc.collect()

    # --- STEP 6: RECOVERY ---
    infovec = np.array([[jmax_row - 1], [jmax_col - 1], [0], [0]])

    for _ in range(jmax_row + jmax_col - 2):
        newinfovec = []
        for h in range(infovec.shape[1]):
            m, n, p, q = infovec[:, h]
            decision = TAG_tf[m, n][p, q]
            
            # Use level-specific mappings during recovery
            t2i_row_m, i2t_row_m = MAPS_R[m]
            t2i_col_n, i2t_col_n = MAPS_C[n]

            if decision in [0, 1]: # Column Path
                j_t, k_t, l_t = i2t_col_n[q]
                t2i_prev, _ = MAPS_C[n-1]
                if decision == 0: # Time/Spatial
                    q1, q2 = t2i_prev.get((j_t+1, 2*k_t, l_t)), t2i_prev.get((j_t+1, 2*k_t+1, l_t))
                else: # Frequency
                    q1, q2 = t2i_prev.get((j_t, k_t, 2*l_t)), t2i_prev.get((j_t, k_t, 2*l_t+1))
                if q1 is not None: newinfovec.append([m, n-1, p, q1])
                if q2 is not None: newinfovec.append([m, n-1, p, q2])

            elif decision in [2, 3]: # Row Path
                j_t, k_t, l_t = i2t_row_m[p]
                t2i_prev, _ = MAPS_R[m-1]
                if decision == 2: # Time/Spatial
                    p1, p2 = t2i_prev.get((j_t+1, 2*k_t, l_t)), t2i_prev.get((j_t+1, 2*k_t+1, l_t))
                else: # Frequency
                    p1, p2 = t2i_prev.get((j_t, k_t, 2*l_t)), t2i_prev.get((j_t, k_t, 2*l_t+1))
                if p1 is not None: newinfovec.append([m-1, n, p1, q])
                if p2 is not None: newinfovec.append([m-1, n, p2, q])

        infovec = np.array(newinfovec).T

    row_indices = infovec[2, :].astype(int)
    col_indices = infovec[3, :].astype(int)
    best_basis_coeffs = dmatrix[row_indices, col_indices]
    
    return best_basis_coeffs, infovec[2:4, :], GProws, GPcols

def ghwt_2d_sparse(A, threshold, rs_row,rs_col,row_order,col_order):
    """
    Performs sparse approximation of a 2D matrix using the EGHWT Best Basis.
    
    Parameters:
    A: 2D NumPy array (the original signal/image).
    threshold: Energy threshold (e.g., 0.95 for 95% energy).
    GProws, GPcols: GraphPart objects for rows and columns.
    
    Returns:
    dvec_rec: The retained (sparse) coefficients.
    infovec_rec: The (row_idx, col_idx) for those coefficients.
    """
    dvec, infovec, GProws, GPcols = ghwt_tf_bestbasis_2d(A, rs_row,rs_col,row_order,col_order)
    
    # Ensure dvec is a flat 1D array for sorting
    A_flat = dvec.flatten()
    
    # 2. Sort coefficients by magnitude in descending order
    # Julia's sortperm(rev=true)
    sorted_indices = np.argsort(np.abs(A_flat))[::-1]
    
    # 3. Calculate Cumulative Energy Percentage
    # E = cumsum(|x|^2) / sum(|x|^2)
    squared_coeffs = np.abs(A_flat[sorted_indices])**2
    total_energy = np.sum(squared_coeffs)
    
    if total_energy == 0:
        return np.array([]), np.array([])
        
    E = np.cumsum(squared_coeffs) / total_energy
    
    # 4. Find how many coefficients are needed to exceed the energy threshold
    n_coeff = np.searchsorted(E, threshold) + 1
    
    # 5. Extract the sparse representation
    dvec_rec = A_flat[sorted_indices[:n_coeff]]
    
    # infovec contains the spatial (row, col) coordinates
    # We slice the columns based on the top indices
    infovec_rec = infovec[:, sorted_indices[:n_coeff]]
    
    return dvec_rec, infovec_rec, GProws, GPcols

def walsh_multiplication_fast(signal, dvec_rec, infovec_rec, GProws, GPcols):
    """
    Applies a Walsh-basis filter to a 1D signal.
    """
    dm_x,_ = ghwt_analysis(signal, GPcols.rs, GPcols.ind)
    dm_x_flat = dm_x.flatten(order='F')
    col_indices = infovec_rec[1, :].astype(int)
    dm_x_selected = dm_x_flat[col_indices]
    
    # 2. Perform the multiplication (Filtering in the Walsh domain)
    L = dm_x_selected * dvec_rec
    
    # 3. Map filtered coefficients back to a row dictionary structure
    m, jmax_row = GProws.tag.shape
    dm_g = np.zeros(m * jmax_row)
    
    for i in range(len(L)):
        row_idx = int(infovec_rec[0, i])
        dm_g[row_idx] += L[i]
    
    # Reshape back to (Space, Levels)
    dm_g = dm_g.reshape((m, jmax_row), order='F')
    
    # 4. Synthesize back to the vertex domain
    f_rec = ghwt_synthesis_aftermultiplication(dm_g, GProws)
    return f_rec

def ghwt_synthesis_aftermultiplication(dmatrix, GP):
    """
    Coarse-to-fine synthesis for GHWT.
    dmatrix: 2D array (N, jmax) - Note: needs to be expanded to 3D for math logic
    GP: GraphPart object
    """
    rs = GP.rs
    tag = GP.tag
    jmax = len(rs)
    N = dmatrix.shape[0]
    
    # Add a dummy 3rd dimension if it's 2D to keep math consistent with 1D analysis
    if dmatrix.ndim == 2:
        dmatrix = dmatrix[:, :, np.newaxis]
        
    # Iterate from coarsest (j=0) to finest (j=jmax-2)
    for j in range(jmax - 1):
        regioncount = len(rs[j]) - 1
        
        for r in range(regioncount):
            rs1 = rs[j][r] - 1
            rs3 = rs[j][r + 1] - 1
            n = rs3 - rs1
            
            if n == 1:
                # Single node region: Copy to the next level
                dmatrix[rs1, j + 1, :] += dmatrix[rs1, j, :]
                
            elif n > 1:
                # Find the split point rs2
                rs2 = rs1 + 1
                while rs2 < rs3 and tag[rs2, j + 1] != 0:
                    rs2 += 1
                
                if rs2 == rs3:
                    # Parent is a copy of the subregion
                    dmatrix[rs1:rs3, j + 1, :] += dmatrix[rs1:rs3, j, :]
                else:
                    # Parent has 2 child regions
                    n1 = rs2 - rs1
                    n2 = rs3 - rs2
                    sqrt_n = np.sqrt(n)
                    sqrt_n1 = np.sqrt(n1)
                    sqrt_n2 = np.sqrt(n2)
                    
                    # INVERSE SCALING & HAAR (The "Butterfly" operation)
                    # child1 (scaling)
                    dmatrix[rs1, j + 1, :] += (
                        sqrt_n1 * dmatrix[rs1, j, :] + 
                        sqrt_n2 * dmatrix[rs1 + 1, j, :]
                    ) / sqrt_n
                    
                    # child2 (difference)
                    dmatrix[rs2, j + 1, :] += (
                        sqrt_n2 * dmatrix[rs1, j, :] - 
                        sqrt_n1 * dmatrix[rs1 + 1, j, :]
                    ) / sqrt_n
                    
                    # INVERSE WALSH
                    parent = rs1 + 2
                    child1 = rs1 + 1
                    child2 = rs2 + 1
                    sqrt2 = np.sqrt(2)
                    
                    while child1 < rs2 or child2 < rs3:
                        # Determine which child should receive the coefficient
                        # logic matches Julia's tag-based sorting
                        if child2 == rs3 or (child1 < rs2 and tag[child1, j+1] < tag[child2, j+1]):
                            dmatrix[child1, j + 1, :] += dmatrix[parent, j, :]
                            child1 += 1
                            parent += 1
                        elif child1 == rs2 or (child2 < rs3 and tag[child2, j+1] < tag[child1, j+1]):
                            dmatrix[child2, j + 1, :] += dmatrix[parent, j, :]
                            child2 += 1
                            parent += 1
                        else:
                            # Sum and difference for matching tags
                            val_p = dmatrix[parent, j, :]
                            val_p_next = dmatrix[parent + 1, j, :]
                            
                            dmatrix[child1, j + 1, :] += (val_p + val_p_next) / sqrt2
                            dmatrix[child2, j + 1, :] += (val_p - val_p_next) / sqrt2
                            
                            child1 += 1
                            child2 += 1
                            parent += 2
                            
    # Final level is the reconstructed signal in partition order
    f_permuted = dmatrix[:, jmax - 1, 0]
    # Revert permutation to get back to original vertex order
    invID = np.argsort(GP.ind) 
    
    return f_permuted[invID]


def ghwt_tf_synthesis_2d_core(dmatrix, tag, rs):
    """
    Core synthesis function operating on a 3D array (N, jmax, num_signals).
    Synthesizes from coarse to fine levels.

    Parameters:
    dmatrix: 3D numpy array (N, jmax, num_signals) — modified in place
    tag: 2D numpy array (N, jmax)
    rs: list of lists, rs[j] contains 1-based boundaries for level j
    """
    jmax = tag.shape[1]

    for j in range(jmax - 1):
        regioncount = len(rs[j]) - 1

        for r in range(regioncount):
            rs1 = rs[j][r] - 1       # convert to 0-based
            rs3 = rs[j][r + 1] - 1
            n = rs3 - rs1

            # Skip if all coefficients are zero in this region at this level
            if not np.any(dmatrix[rs1:rs3, j, :]):
                continue

            if n == 1:
                dmatrix[rs1, j + 1, :] += dmatrix[rs1, j, :]

            elif n > 1:
                rs2 = rs1 + 1
                while rs2 < rs3 and tag[rs2, j + 1] != 0:
                    rs2 += 1

                if rs2 == rs3:
                    # Parent is a copy of the subregion
                    dmatrix[rs1:rs3, j + 1, :] += dmatrix[rs1:rs3, j, :]
                else:
                    n1 = rs2 - rs1
                    n2 = rs3 - rs2
                    sqrt_n = np.sqrt(n)
                    sqrt_n1 = np.sqrt(n1)
                    sqrt_n2 = np.sqrt(n2)

                    # Inverse scaling & Haar
                    dmatrix[rs1, j + 1, :] += (
                        sqrt_n1 * dmatrix[rs1, j, :] +
                        sqrt_n2 * dmatrix[rs1 + 1, j, :]
                    ) / sqrt_n

                    dmatrix[rs2, j + 1, :] += (
                        sqrt_n2 * dmatrix[rs1, j, :] -
                        sqrt_n1 * dmatrix[rs1 + 1, j, :]
                    ) / sqrt_n

                    # Inverse Walsh
                    parent = rs1 + 2
                    child1 = rs1 + 1
                    child2 = rs2 + 1
                    sqrt2 = np.sqrt(2)

                    while child1 < rs2 or child2 < rs3:
                        if child2 == rs3 or (child1 < rs2 and tag[child1, j + 1] < tag[child2, j + 1]):
                            dmatrix[child1, j + 1, :] += dmatrix[parent, j, :]
                            child1 += 1
                            parent += 1
                        elif child1 == rs2 or (child2 < rs3 and tag[child2, j + 1] < tag[child1, j + 1]):
                            dmatrix[child2, j + 1, :] += dmatrix[parent, j, :]
                            child2 += 1
                            parent += 1
                        else:
                            val_p = dmatrix[parent, j, :].copy()
                            val_p1 = dmatrix[parent + 1, j, :].copy()
                            dmatrix[child1, j + 1, :] += (val_p + val_p1) / sqrt2
                            dmatrix[child2, j + 1, :] += (val_p - val_p1) / sqrt2
                            child1 += 1
                            child2 += 1
                            parent += 2

    return dmatrix


def ghwt_tf_synthesis_2d(dvec, infovec, GProws, GPcols):
    """
    Synthesize a matrix from best-basis coefficients.

    Parameters:
    dvec: 1D array of best-basis coefficients
    infovec: 2×K array where row 0 = row indices, row 1 = col indices (0-based)
    GProws: GraphPart for rows
    GPcols: GraphPart for columns

    Returns:
    matrix: reconstructed 2D matrix (in original ordering)
    """
    tag_col, rs_col = GPcols.tag, GPcols.rs
    tag_row, rs_row = GProws.tag, GProws.rs
    fcols, jmax_col = tag_col.shape
    frows, jmax_row = tag_row.shape

    # 1. Place coefficients into the full 2D dictionary matrix
    dmatrix = np.zeros((frows * jmax_row, fcols * jmax_col))
    for i in range(infovec.shape[1]):
        dmatrix[int(infovec[0, i]), int(infovec[1, i])] = dvec[i]

    # 2. Column-direction synthesis
    #    Julia: reshape(dmatrix', fcols, jmax_col, frows*jmax_row) — column-major
    matrix_r = dmatrix.T  # (fcols*jmax_col, frows*jmax_row)
    matrix_r = matrix_r.reshape((fcols, jmax_col, frows * jmax_row), order='F')
    matrix_r = ghwt_tf_synthesis_2d_core(matrix_r, tag_col, rs_col)
    matrix_r = matrix_r[:, -1, :]  # (fcols, frows*jmax_row) — take finest level

    # 3. Row-direction synthesis
    #    Julia: reshape(matrix_r', frows, jmax_row, fcols) — column-major
    matrix_r = matrix_r.T  # (frows*jmax_row, fcols)
    matrix_r = matrix_r.reshape((frows, jmax_row, fcols), order='F')
    matrix_r = ghwt_tf_synthesis_2d_core(matrix_r, tag_row, rs_row)
    matrix_r = matrix_r[:, -1, :]  # (frows, fcols) — take finest level

    # 4. Undo permutation to original ordering
    matrix = np.zeros((frows, fcols))
    matrix[np.ix_(GProws.ind, GPcols.ind)] = matrix_r

    return matrix