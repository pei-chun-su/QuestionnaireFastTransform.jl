import cupy as cp
import numpy as np
import multiprocessing as mp
import torch
def cosine_tile_worker(gpu_id, data_cpu, row_start, row_end, tile_size, take_abs, threshold):
    """
    Worker function to compute a horizontal slice of the similarity matrix using PyTorch.
    Computes only the upper triangle (j >= i) to save time.
    """
    try:
        device = torch.device(f"cuda:{gpu_id}")
        with torch.cuda.device(device):
            n_cols = data_cpu.shape[1]
            num_rows_in_slice = row_end - row_start
            
            # Resulting slice for this GPU
            slice_cpu = np.zeros((num_rows_in_slice, n_cols), dtype=np.float32)
            
            # 1. Transfer data to GPU
            data_gpu_all = torch.from_numpy(data_cpu).to(device)
            
            # Pre-calculate norms for all columns
            all_norms = torch.linalg.norm(data_gpu_all, dim=0)
            all_norms[all_norms == 0] = 1.0
            data_norm_all = data_gpu_all / all_norms

            # 2. Iterate through tiles of assigned rows
            for i in range(row_start, row_end, tile_size):
                i_end = min(i + tile_size, row_end)
                # (tile_size x m_clean)
                tile_i = data_norm_all[:, i:i_end].t()
                
                # 3. Iterate through tiles of columns (j >= i for symmetry)
                for j in range(i, n_cols, tile_size):
                    j_end = min(j + tile_size, n_cols)
                    tile_j = data_norm_all[:, j:j_end] # (m_clean x tile_size)
                    
                    # Compute dot product (Cosine Similarity)
                    sim_gpu = torch.mm(tile_i, tile_j)
                    
                    # Post-process on GPU
                    if take_abs:
                        sim_gpu = torch.abs(sim_gpu)
                    else:
                        sim_gpu = torch.clamp(sim_gpu, min=0.0)
                    
                    if threshold > 0.0:
                        sim_gpu[sim_gpu < threshold] = 0.0
                    
                    # Store in slice (local indexing)
                    local_i_start = i - row_start
                    local_i_end = i_end - row_start
                    slice_cpu[local_i_start:local_i_end, j:j_end] = sim_gpu.cpu().numpy()
                
                # Manual memory cleanup
                torch.cuda.empty_cache()
                
            return row_start, row_end, slice_cpu
            
    except Exception as e:
        return (gpu_id, str(e))

def gaussian_tile_worker(gpu_id, data_clean, r_start, r_end, knn, medians=None):
    """
    Worker function to be used by Pool.starmap
    """
    try:
        cp.cuda.Device(gpu_id).use()
        
        # Move data slice to GPU (or full data if it fits)
        # For Euclidean, we need the full data_clean to compare against r_start:r_end
        d_gpu = cp.array(data_clean) 
        row_slice = d_gpu[r_start:r_end, :]
        
        # Squared Euclidean using: ||u-v||^2 = ||u||^2 + ||v||^2 - 2<u,v>
        # row_slice: (m, d), d_gpu: (N, d)
        dot_product = row_slice @ d_gpu.T  # (m, N)
        u_sq = cp.sum(row_slice**2, axis=1, keepdims=True) # (m, 1)
        v_sq = cp.sum(d_gpu**2, axis=1) # (N,)
        
        dist_sq = cp.maximum(u_sq + v_sq - 2 * dot_product, 0)

        # MODE 1: Calculate Medians (KNN)
        if medians is None:
            # We only need the smallest knn distances for these rows
            knn_dists_sq = cp.partition(dist_sq, knn, axis=1)[:, :knn]
            return (r_start, r_end, cp.asnumpy(cp.sqrt(knn_dists_sq)))

        # MODE 2: Calculate Gaussian Affinity
        else:
            m_slice = cp.array(medians[r_start:r_end, np.newaxis])
            m_slice[m_slice == 0] = 1e-15
            
            affinity_slice = cp.exp(-(dist_sq / (m_slice**2)))
            return (r_start, r_end, cp.asnumpy(affinity_slice))

    except Exception as e:
        return (gpu_id, str(e))

def memory_lean_tile_worker(gpu_id, f_cpu, nlevels, partition_list, alpha, removemean, row_start, row_end, tile_size=1000):
    with cp.cuda.Device(gpu_id):
        p, n, q = f_cpu.shape
        W_slice_cpu = np.zeros((row_end - row_start, n), dtype=np.float32)
        
        # Move ONLY the rows this GPU is responsible for to VRAM
        # This is the "Horizontal Slice" (e.g., 200MB)
        newpoints_tile_gpu = cp.array(f_cpu[:, row_start:row_end, :], dtype=cp.float32)

        # Instead of moving all f, we stream f in "Column Tiles"
        for col_start in range(0, n, tile_size):
            col_end = min(col_start + tile_size, n)
            
            # Move only a small vertical chunk of f to GPU (e.g., 100MB)
            points_chunk_gpu = cp.array(f_cpu[:, col_start:col_end, :], dtype=cp.float32)
            
            for idx, i in enumerate(nlevels):
                partition = cp.array(partition_list[idx])
                # ... perform math between newpoints_tile_gpu and points_chunk_gpu ...
                # result = mat2_flat.T @ mat_flat 
                
                # Store result in the correct columns of W_slice_cpu
                # W_slice_cpu[:, col_start:col_end] += result.get()

            del points_chunk_gpu
            cp.get_default_memory_pool().free_all_blocks()
            
        return row_start, row_end, W_slice_cpu

def local_geometry_tile_worker(gpu_id, f_cpu, nlevels, partition_list, alpha, removemean, r_start, r_end):
    try:
        device = torch.device(f"cuda:{gpu_id}")
        with torch.cuda.device(device):
            # 1. Prepare data in VRAM
            # Convert numpy to torch tensor and move to GPU
            f_gpu = torch.from_numpy(f_cpu).to(device)
            num_rows = r_end - r_start
            n_total = f_cpu.shape[1]
            q = f_cpu.shape[2]
            
            W_tile_gpu = torch.zeros((num_rows, n_total), dtype=torch.float32, device=device)
            tile_data_gpu = f_gpu[:, r_start:r_end, :]

            for idx, i in enumerate(nlevels):
                part_gpu = torch.from_numpy(np.array(partition_list[idx])).to(device)
                fold_count = int(part_gpu.max() + 1)
                
                level_sim = torch.zeros((num_rows, n_total), dtype=torch.float32, device=device)
                cI = 0
                
                for fold in range(fold_count):
                    # Find indices where partition matches
                    I = (part_gpu == fold).nonzero(as_tuple=True)[0]
                    if len(I) <= 1: continue
                    
                    partition_weight = len(I)
                    cI += partition_weight
                    
                    mat_ref = f_gpu[I]   # [p_fold, n_total, q]
                    mat_tile = tile_data_gpu[I] # [p_fold, num_rows, q]
                    
                    # --- A. MEAN REMOVAL ---
                    if removemean:
                        # nanmean equivalent in torch (using nanmean for 1.10+)
                        mat_ref -= torch.nanmean(mat_ref, dim=0, keepdim=True)
                        mat_tile -= torch.nanmean(mat_tile, dim=0, keepdim=True)
                    
                    # --- B. NORMALIZATION ---
                    norm_ref = torch.linalg.norm(mat_ref, dim=0) + 1e-12
                    norm_tile = torch.linalg.norm(mat_tile, dim=0) + 1e-12
                    
                    mat_ref /= norm_ref.unsqueeze(0)
                    mat_tile /= norm_tile.unsqueeze(0)
                    
                    # --- C. MATRIX MULTIPLICATION ---
                    # Permute and reshape to (pq, samples)
                    # transpose(0,2,1) in Cupy -> permute(0, 2, 1) in Torch
                    ref_flat = mat_ref.permute(0, 2, 1).reshape(-1, n_total)
                    tile_flat = mat_tile.permute(0, 2, 1).reshape(-1, num_rows)
                    
                    # Core math: tile_flat.T (num_rows, pq) @ ref_flat (pq, n_total)
                    level_sim += torch.abs(torch.mm(tile_flat.t(), ref_flat)) * partition_weight
                
                # --- D. LEVEL WEIGHTING ---
                if cI > 0:
                    level_sim /= (q * cI)
                    W_tile_gpu += (2.0**(alpha * (1.0 - i))) * level_sim

            # Return the processed slice to CPU as numpy
            res = W_tile_gpu.cpu().numpy()
            
            # Explicit Cleanup
            del f_gpu, tile_data_gpu, W_tile_gpu
            torch.cuda.empty_cache()
            
            return r_start, r_end, res
            
    except Exception as e:
        return (gpu_id, str(e))

def emd_tile_worker(gpu_id, ext_vecs_cpu, c_start, c_end):
    """
    Worker for calculating slices of the Cityblock (L1) distance matrix.
    ext_vecs_cpu: (tree_size x num_columns) weighted embeddings.
    """
    try:
        cp.cuda.Device(gpu_id).use()
        
        # Move the full embedding and the specific column slice to GPU
        # ext_vecs_cpu is (D, N), where D is tree size and N is num columns
        full_ext = cp.array(ext_vecs_cpu, dtype=cp.float32)
        slice_ext = full_ext[:, c_start:c_end] 
        
        # L1 distance: sum(|A - B|)
        # Use broadcasting: slice_ext is (D, chunk), full_ext is (D, N)
        # To avoid (D, chunk, N) memory blowup, we use a loop or a custom kernel
        num_cols = full_ext.shape[1]
        chunk_size = c_end - c_start
        dist_slice = cp.zeros((chunk_size, num_cols), dtype=cp.float32)

        # Sub-tiling to prevent VRAM overflow during broadcasting
        sub_tile = 500 
        for i in range(0, num_cols, sub_tile):
            end_i = min(i + sub_tile, num_cols)
            
            # target_block will be (chunk_size, sub_tile)
            # We initialize it to zero and add to it iteratively
            acc_diff = cp.zeros((chunk_size, end_i - i), dtype=cp.float32)
            
            # Instead of a 3D broadcast (D, chunk, sub_tile), 
            # we loop over D (the tree dimension)
            # This reduces memory from (D * chunk * sub_tile) to (chunk * sub_tile)
            for d in range(full_ext.shape[0]):
                # Get one row (one tree node) across the chunk and the sub_tile
                val_slice = slice_ext[d, :, cp.newaxis] # (chunk, 1)
                val_full = full_ext[d, cp.newaxis, i:end_i] # (1, sub_tile)
                
                # Absolute difference for this specific tree node
                acc_diff += cp.abs(val_slice - val_full)
            
            dist_slice[:, i:end_i] = acc_diff
            
            # Help garbage collection
            del acc_diff

        return (c_start, c_end, cp.asnumpy(dist_slice))

    except Exception as e:
        return (gpu_id, str(e))