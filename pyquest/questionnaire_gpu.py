"""
questionnaire.py: Main module for running the questionnaire with 
                  prespecified options. Define parameters in a 
                  PyQuestParams object, and then run pyquest(data,params).
"""

import datetime
import affinity_gpu2 as affinity
import dual_affinity_gpu2 as dual_affinity
import bin_tree_build
import flex_tree_build 
import tree_util_gpu as tree_util
import numpy as np
import cupy as cp

INIT_AFF_COS_SIM = 0
INIT_AFF_GAUSSIAN = 1

DEFAULT_INIT_AFF_THRESHOLD = 0.0
DEFAULT_INIT_AFF_EPSILON = 1.0
DEFAULT_INIT_AFF_KNN = 5

TREE_TYPE_BINARY = 0
TREE_TYPE_FLEXIBLE = 1

DEFAULT_TREE_BAL_CONSTANT = 1.0
DEFAULT_TREE_CONSTANT = 1.0

DUAL_EMD = 2
DUAL_GAUSSIAN = 3
DUAL_CORRELATION = 4
DUAL_COSINE = 5

DEFAULT_DUAL_EPSILON = 1.0
DEFAULT_DUAL_ALPHA = 0.0
DEFAULT_DUAL_BETA = 1.0

DEFAULT_N_ITERS = 3
DEFAULT_N_TREES = 1

DEFAULT_WEIGHTED = False
DEFAULT_CUT = 'r_dyadic'

class PyQuestParams(object):
    
    def __init__(self,init_aff_type,tree_type,dual_row_type,dual_col_type,
                 **kwargs):
        
        self.set_init_aff(init_aff_type,**kwargs)
        self.set_tree_type(tree_type,**kwargs)
        self.set_dual_aff(dual_row_type,dual_col_type,**kwargs)
        self.set_iters(**kwargs)
        self.set_gpu(**kwargs)
    
    def set_init_aff(self,affinity_type,**kwargs):
        self.init_aff_type = affinity_type
        if "cut_type" in kwargs:
                self.cut = kwargs["cut_type"]
        else:
                self.cut = DEFAULT_CUT
                
        if self.init_aff_type == INIT_AFF_COS_SIM:
            if "threshold" in kwargs:
                self.init_aff_threshold = kwargs["threshold"]
            else:
                self.init_aff_threshold = DEFAULT_INIT_AFF_THRESHOLD
                
        elif self.init_aff_type == INIT_AFF_GAUSSIAN:
            if "epsilon" in kwargs:
                self.init_aff_epsilon = kwargs["epsilon"]
            else:
                self.init_aff_epsilon = DEFAULT_INIT_AFF_EPSILON                  
            if "knn" in kwargs:
                self.init_aff_knn = kwargs["knn"]
            else:
                self.init_aff_knn = DEFAULT_INIT_AFF_KNN
        
    def set_tree_type(self,tree_type,**kwargs):
        if "diag_bias" in kwargs:
            self.diag_bias = kwargs["diag_bias"]
        else:
            self.diag_bias = False
        if type(tree_type) is list:
            self.row_tree_type = tree_type[0]
            if self.row_tree_type == TREE_TYPE_BINARY:
                if "row_bal_constant" in kwargs:
                    self.row_tree_bal_constant = kwargs["row_bal_constant"]
                else:
                    self.row_tree_bal_constant = DEFAULT_TREE_BAL_CONSTANT

            if self.row_tree_type == TREE_TYPE_FLEXIBLE:
                if "row_tree_constant" in kwargs:
                    self.row_tree_constant = kwargs["row_tree_constant"]
                else:
                    self.row_tree_constant = DEFAULT_TREE_CONSTANT

            self.col_tree_type = tree_type[1]
            if self.col_tree_type == TREE_TYPE_BINARY:
                if "col_bal_constant" in kwargs:
                    self.col_tree_bal_constant = kwargs["col_bal_constant"]
                else:
                    self.col_tree_bal_constant = DEFAULT_TREE_BAL_CONSTANT

            if self.col_tree_type == TREE_TYPE_FLEXIBLE:
                if "col_tree_constant" in kwargs:
                    self.col_tree_constant = kwargs["col_tree_constant"]
                else:
                    self.col_tree_constant = DEFAULT_TREE_CONSTANT
        else:
            self.row_tree_type = tree_type
            self.col_tree_type = tree_type
            if tree_type == TREE_TYPE_BINARY:
                if "bal_constant" in kwargs:
                    self.row_tree_bal_constant = kwargs["bal_constant"]
                    self.col_tree_bal_constant = kwargs["bal_constant"]
                else:
                    self.row_tree_bal_constant = DEFAULT_TREE_BAL_CONSTANT
                    self.col_tree_bal_constant = DEFAULT_TREE_BAL_CONSTANT

            if tree_type == TREE_TYPE_FLEXIBLE:
                if "tree_constant" in kwargs:
                    self.row_tree_constant = kwargs["tree_constant"]
                    self.col_tree_constant = kwargs["tree_constant"]
                else:
                    self.row_tree_constant = DEFAULT_TREE_CONSTANT
                    self.col_tree_constant = DEFAULT_TREE_CONSTANT
        
    def set_dual_aff(self,row_affinity_type,col_affinity_type,**kwargs):
        self.row_affinity_type = row_affinity_type
        self.col_affinity_type = col_affinity_type
        
        if self.row_affinity_type == DUAL_GAUSSIAN:
            if "row_epsilon" in kwargs:
                self.row_epsilon = kwargs["row_epsilon"]
            else:
                self.row_epsilon = DEFAULT_DUAL_EPSILON                  
                
        if self.row_affinity_type == DUAL_EMD:
            if "row_alpha" in kwargs:
                self.row_alpha = kwargs["row_alpha"]
            else:
                self.row_alpha = DEFAULT_DUAL_ALPHA
            if "row_beta" in kwargs:
                self.row_beta = kwargs["row_beta"]
            else:
                self.row_beta = DEFAULT_DUAL_BETA
            if "row_weighted" in kwargs:
                self.row_weighted = kwargs["row_weighted"]
            else:
                self.row_weighted = DEFAULT_WEIGHTED
                
        if self.row_affinity_type == DUAL_CORRELATION:
            if "row_alpha" in kwargs:
                self.row_alpha = kwargs["row_alpha"]
            else:
                self.row_alpha = DEFAULT_DUAL_ALPHA
            if "row_beta" in kwargs:
                self.row_beta = kwargs["row_beta"]
            else:
                self.row_beta = DEFAULT_DUAL_BETA
            if "row_weighted" in kwargs:
                self.row_weighted = kwargs["row_weighted"]
            else:
                self.row_weighted = DEFAULT_WEIGHTED
                
        if self.row_affinity_type == DUAL_COSINE:
            if "row_alpha" in kwargs:
                self.row_alpha = kwargs["row_alpha"]
            else:
                self.row_alpha = DEFAULT_DUAL_ALPHA
            if "row_beta" in kwargs:
                self.row_beta = kwargs["row_beta"]
            else:
                self.row_beta = DEFAULT_DUAL_BETA
            if "row_weighted" in kwargs:
                self.row_weighted = kwargs["row_weighted"]
            else:
                self.row_weighted = DEFAULT_WEIGHTED
            
        
        if self.col_affinity_type == DUAL_GAUSSIAN:
            if "col_epsilon" in kwargs:
                self.col_epsilon = kwargs["col_epsilon"]
            else:
                self.col_epsilon = DEFAULT_DUAL_EPSILON                  
                
        if self.col_affinity_type == DUAL_EMD:
            if "col_alpha" in kwargs:
                self.col_alpha = kwargs["col_alpha"]
            else:
                self.col_alpha = DEFAULT_DUAL_ALPHA
            if "col_beta" in kwargs:
                self.col_beta = kwargs["col_beta"]
            else:
                self.col_beta = DEFAULT_DUAL_BETA
            if "col_weighted" in kwargs:
                self.col_weighted = kwargs["col_weighted"]
            else:
                self.col_weighted = DEFAULT_WEIGHTED
                
        if self.col_affinity_type == DUAL_CORRELATION:
            if "col_alpha" in kwargs:
                self.col_alpha = kwargs["col_alpha"]
            else:
                self.col_alpha = DEFAULT_DUAL_ALPHA
            if "col_beta" in kwargs:
                self.col_beta = kwargs["col_beta"]
            else:
                self.col_beta = DEFAULT_DUAL_BETA
            if "col_weighted" in kwargs:
                self.col_weighted = kwargs["col_weighted"]
            else:
                self.col_weighted = DEFAULT_WEIGHTED 
                
        if self.col_affinity_type == DUAL_COSINE:
            if "col_alpha" in kwargs:
                self.col_alpha = kwargs["col_alpha"]
            else:
                self.col_alpha = DEFAULT_DUAL_ALPHA
            if "col_beta" in kwargs:
                self.col_beta = kwargs["col_beta"]
            else:
                self.col_beta = DEFAULT_DUAL_BETA
            if "col_weighted" in kwargs:
                self.col_weighted = kwargs["col_weighted"]
            else:
                self.col_weighted = DEFAULT_WEIGHTED 
            
                
        

    def set_iters(self,**kwargs):
        if "n_iters" in kwargs:
            self.n_iters = kwargs["n_iters"]
        else:
            print("default n_iters")
            self.n_iters = DEFAULT_N_ITERS
        
        if "n_trees" in kwargs:
            self.n_trees = kwargs["n_trees"]
        else:
            self.n_trees = DEFAULT_N_TREES
    def set_gpu(self,**kwargs):
        if "ngpu" in kwargs:
            self.ngpu = kwargs["ngpu"]
        else:
            print("default ngpu = 1")
            self.ngpu = 1
        if "ntile" in kwargs:
            self.ntile = kwargs["ntile"]
        else:
            print("default ntile = 2000")
            self.ntile = 2000
    

class PyQuestRun(object):
    """
    Holds the results of a run of the questionnaire, which are basically:
    a description of when the run was done, the trees which were generated on
    each iteration, and the parameters.
    """
    def __init__(self,run_desc,row_trees,col_trees,row_tree_descs,
                 col_tree_descs,params,row_aff,col_aff):
        self.run_desc = run_desc
        self.row_trees = row_trees
        self.col_trees = col_trees
        self.row_tree_descs = row_tree_descs
        self.col_tree_descs = col_tree_descs
        self.params = params
        self.col_aff = col_aff
        self.row_aff = row_aff


def pyquest(data,params):
    """
    Runs the questionnaire on data with params. params is a PyQuestParams object.
	Starts by constructing the initial affinity on the rows of the matrix (default).
    """
    
    # construct row affinity
    if params.init_aff_type == INIT_AFF_COS_SIM:
        init_row_aff = affinity.mutual_cosine_similarity_tiled_parallel(
                            data.T,False,None,params.init_aff_threshold,params.ntile,params.ngpu)
    elif params.init_aff_type == INIT_AFF_GAUSSIAN:
        init_row_aff = affinity.gaussian_euclidean_tiled_parallel(data.T, params.init_aff_knn, params.init_aff_epsilon, params.ntile,params.ngpu)
                            
    
    #Initial row tree
    if params.row_tree_type == TREE_TYPE_BINARY:
        init_row_tree = bin_tree_build.bin_tree_build(init_row_aff, params.cut,
                                                      params.row_tree_bal_constant,params.diag_bias)
    elif params.row_tree_type == TREE_TYPE_FLEXIBLE:
        init_row_tree = flex_tree_build.flex_tree_diffusion(init_row_aff,
                                            params.row_tree_constant)
                                            
    # data structure for trees. All trees calculated in the process are exported                                           
    dual_row_trees = [init_row_tree]
    dual_col_trees = []
    
    row_tree_descs = ["Initial tree"]
    col_tree_descs = []
    
    # iterate over the questionnaire starting with columns and then rows in each iteration
    for i in range(params.n_iters):
        message = "Iteration {}".format(i)
        print(message)
        
        # calculating column affinity based on row tree
        #print "Beginning iteration {}".format(i)
        if params.col_affinity_type == DUAL_EMD:
            print("emd distance.")
            if params.col_weighted == True:
                print("weighted emd")
                row_coefs = tree_util.tree_transform_mat(dual_row_trees[-1]).dot(data)
                row_weights = np.sqrt(np.sum(row_coefs**2,axis = 1))
                col_emd = dual_affinity.calc_emd_tiled_parallel(data,dual_row_trees[-1],
                      alpha=0,beta=0,weights=row_weights,num_gpus = params.ngpu, task_width=params.ntile)
            else:
                col_emd = dual_affinity.calc_emd_tiled_parallel(data,dual_row_trees[-1],
                     params.col_alpha,params.col_beta, num_gpus= params.ngpu,task_width=params.ntile)
            col_aff = dual_affinity.emd_dual_aff(col_emd)
        elif params.col_affinity_type == DUAL_GAUSSIAN:
            print("Gaussian dual affinity not supported at the moment.")
            return None
        elif params.col_affinity_type == DUAL_CORRELATION:
            print("correlation distance.")
            col_aff = dual_affinity.partition_dualgeometry_tiled_parallel(np.expand_dims(data, axis=2), dual_row_trees[-1], params.col_alpha, 
                                                                          1,params.ngpu,params.ntile)
        elif params.col_affinity_type == DUAL_COSINE:
            print("cosine distance.")
            col_aff = dual_affinity.partition_dualgeometry_tiled_parallel(np.expand_dims(data, axis=2), dual_row_trees[-1], params.col_alpha, 
                                                                          0,params.ngpu,params.ntile)
           
        
        # constructing column tree
        if params.col_tree_type == TREE_TYPE_BINARY:
            col_tree = bin_tree_build.bin_tree_build(col_aff,params.cut,
                                                     params.col_tree_bal_constant,params.diag_bias)
        elif params.col_tree_type == TREE_TYPE_FLEXIBLE:
            col_tree = flex_tree_build.flex_tree_diffusion(col_aff,
                                                           params.col_tree_constant)
        dual_col_trees.append(col_tree)
        col_tree_descs.append("Iteration {}".format(i))
        
        
        # calculate row affinity based on column tree
        if params.row_affinity_type == DUAL_EMD:
            print("emd distance.")
            if params.row_weighted == True:
                print("weighted emd")
                col_coefs = tree_util.tree_transform_mat(dual_col_trees[-1]).dot(data.T)
                col_weights = np.sqrt(np.sum(col_coefs**2,axis = 1))
                row_emd = dual_affinity.calc_emd_tiled_parallel(data.T,dual_col_trees[-1],
                      alpha=0,beta=0,weights=col_weights,num_gpus = params.ngpu, task_width=params.ntile)
            else:
                row_emd = dual_affinity.calc_emd_tiled_parallel(data.T,dual_col_trees[-1],
                     params.row_alpha,params.row_beta, num_gpus= params.ngpu,task_width=params.ntile)
            row_aff = dual_affinity.emd_dual_aff(row_emd)
        elif params.row_affinity_type == DUAL_GAUSSIAN:
            print("Gaussian dual affinity not supported at the moment.")
            return None
        elif params.row_affinity_type == DUAL_CORRELATION:
            print("correlation distance.")
            row_aff = dual_affinity.partition_dualgeometry_tiled_parallel(np.expand_dims(data.T, axis=2), dual_col_trees[-1], params.row_alpha,
                                                                          1,params.ngpu,params.ntile)
            #row_aff = dual_affinity.partition_dualgeometry(np.expand_dims(data.T, axis=2), dual_col_trees[-1], params.row_alpha, 1)
        elif params.row_affinity_type == DUAL_COSINE:
            print("cosine distance.")
            row_aff = dual_affinity.partition_dualgeometry_tiled_parallel(np.expand_dims(data.T, axis=2), dual_col_trees[-1], params.row_alpha, 
                                                                          0,params.ngpu,params.ntile)
            #row_aff = dual_affinity.partition_dualgeometry(np.expand_dims(data.T, axis=2), dual_col_trees[-1], params.row_alpha, 0)
            
 
        
        # constructing row tree
        if params.row_tree_type == TREE_TYPE_BINARY:
            row_tree = bin_tree_build.bin_tree_build(row_aff,params.cut,
                                                     params.row_tree_bal_constant,params.diag_bias)
        elif params.row_tree_type == TREE_TYPE_FLEXIBLE:
            row_tree = flex_tree_build.flex_tree_diffusion(row_aff,
                                                           params.row_tree_constant)
        dual_row_trees.append(row_tree)
        row_tree_descs.append("Iteration {}".format(i))
        quest_run_desc = "{}".format(datetime.datetime.now())
	# iterations have finished, outputting structures of the tree, 
    # parameters 
    return PyQuestRun(quest_run_desc,dual_row_trees,dual_col_trees,
                      row_tree_descs,col_tree_descs,params,row_aff,col_aff)


class PyQuest3DParams(PyQuestParams):
    
    def __init__(self,init_aff_type,tree_type,dual_row_type,dual_col_type,dual_chan_type,
                 **kwargs):
        
        self.set_init_aff(init_aff_type,**kwargs)
        self.set_tree_type(tree_type,**kwargs)
        self.set_dual_aff(dual_row_type,dual_col_type,dual_chan_type,**kwargs)
        self.set_iters(**kwargs)
    
    def set_dual_aff(self,row_affinity_type,col_affinity_type,chan_affinity_type,**kwargs):
        
        super(PyQuest3DParams, self).set_dual_aff(row_affinity_type,col_affinity_type,**kwargs)
        self.chan_affinity_type = chan_affinity_type
        if self.chan_affinity_type == DUAL_GAUSSIAN:
            if "chan_epsilon" in kwargs:
                self.chan_epsilon = kwargs["chan_epsilon"]
            else:
                self.chan_epsilon = DEFAULT_DUAL_EPSILON                  
                
        if self.chan_affinity_type == DUAL_EMD:
            if "chan_alpha" in kwargs:
                self.chan_alpha = kwargs["chan_alpha"]
            else:
                self.chan_alpha = DEFAULT_DUAL_ALPHA
            if "chan_beta" in kwargs:
                self.chan_beta = kwargs["chan_beta"]
            else:
                self.chan_beta = DEFAULT_DUAL_BETA
    
    def set_tree_type(self,tree_type,**kwargs):
        super(PyQuest3DParams, self).set_tree_type(tree_type,**kwargs)
        if type(tree_type) is list:
            self.chan_tree_type = tree_type[1]
            if self.chan_tree_type == TREE_TYPE_BINARY:
                if "chan_bal_constant" in kwargs:
                    self.chan_tree_bal_constant = kwargs["chan_bal_constant"]
                else:
                    self.chan_tree_bal_constant = DEFAULT_TREE_BAL_CONSTANT

            if self.chan_tree_type == TREE_TYPE_FLEXIBLE:
                if "chan_tree_constant" in kwargs:
                    self.chan_tree_constant = kwargs["chan_tree_constant"]
                else:
                    self.chan_tree_constant = DEFAULT_TREE_CONSTANT
        else:
            self.chan_tree_type = tree_type
            if tree_type == TREE_TYPE_BINARY:
                if "bal_constant" in kwargs:
                    self.chan_tree_bal_constant = kwargs["bal_constant"]
                else:
                    self.chan_tree_bal_constant = DEFAULT_TREE_BAL_CONSTANT

            if tree_type == TREE_TYPE_FLEXIBLE:
                if "tree_constant" in kwargs:
                    self.chan_tree_constant = kwargs["tree_constant"]
                else:
                    self.chan_tree_constant = DEFAULT_TREE_CONSTANT

class PyQuest3DRun(PyQuestRun):
    """
    Holds the results of a run of the questionnaire, which are basically:
    a description of when the run was done, the trees which were generated on
    each iteration, and the parameters.
    """
    def __init__(self,run_desc,row_trees,col_trees,chan_trees,
                 row_tree_descs,col_tree_descs,chan_tree_descs,params,
                 init_col_aff,init_row_aff,row_aff,col_aff,chan_aff):
        self.run_desc = run_desc
        self.row_trees = row_trees
        self.col_trees = col_trees
        self.chan_trees = chan_trees
        self.row_tree_descs = row_tree_descs
        self.col_tree_descs = col_tree_descs
        self.chan_tree_descs = chan_tree_descs
        self.params = params
        self.init_col_aff = init_col_aff
        self.init_row_aff = init_row_aff
        self.row_aff = row_aff
        self.col_aff = col_aff
        self.chan_aff = chan_aff

def pyquest3d(data3d,params):
    """
    Runs the 3d questionnaire on data with params. 
    params is a PyQuest3DParams object.
	Order of analysis is initialization for rows and columns
	and then iterating over channels (3rd dimension), rows and columns. 
    """
    
    nrows,ncols,nchans = data3d.shape
    data_Y =  np.reshape(data3d, (nrows,ncols*nchans),order='F') 
    data_X =  np.reshape(np.transpose(data3d,(0, 2, 1)), (nrows*nchans, ncols),order='F') 

    if params.init_aff_type == INIT_AFF_COS_SIM:
        init_row_aff = affinity.mutual_cosine_similarity(
                            data_Y.T,False,0,threshold=params.init_aff_threshold)
        init_col_aff = affinity.mutual_cosine_similarity(
                            data_X,False,0,threshold=params.init_aff_threshold)    
    elif params.init_aff_type == INIT_AFF_GAUSSIAN:
        init_row_aff = affinity.gaussian_euclidean(
                            data_Y.T, params.init_aff_knn, params.init_aff_epsilon)
        init_col_aff = affinity.gaussian_euclidean(
                            data_X, params.init_aff_knn, params.init_aff_epsilon)   
    
    #Initial row tree
    if params.row_tree_type == TREE_TYPE_BINARY:
        init_row_tree = bin_tree_build.bin_tree_build(init_row_aff,'r_dyadic',
                                                      params.row_tree_bal_constant)
    elif params.row_tree_type == TREE_TYPE_FLEXIBLE:
        init_row_tree = flex_tree_build.flex_tree_diffusion(init_row_aff,
                                            params.row_tree_constant)
    # initial column tree
    if params.col_tree_type == TREE_TYPE_BINARY:
        init_col_tree = bin_tree_build.bin_tree_build(init_col_aff,'r_dyadic',
                                                      params.col_tree_bal_constant)
    elif params.col_tree_type == TREE_TYPE_FLEXIBLE:
        init_col_tree = flex_tree_build.flex_tree_diffusion(init_col_aff,
                                            params.col_tree_constant)

    # data structure for trees. All trees calculated in the process are exported                                           
    dual_row_trees = [init_row_tree]
    dual_col_trees = [init_col_tree]
    dual_chan_trees = []
    
    row_tree_descs = ["Initial tree"]
    col_tree_descs = ["Initial tree"]
    chan_tree_descs = []
    
    # iterate over the questionnaire starting with channels and then rows and cols in each iteration
    for i in range(params.n_iters):
        message = "Iteration {}: calculating channel affinity...".format(i)

        # calculating channel affinity based on row and col trees
        #print "Beginning iteration {}".format(i)
        if params.chan_affinity_type == DUAL_EMD:
            chan_emd2d = dual_affinity.calc_2demd(data3d, init_row_tree, init_col_tree, 
                                        row_alpha=params.row_alpha, row_beta=params.row_beta,
                                        col_alpha=params.col_alpha, col_beta=params.col_beta)

            chan_aff = dual_affinity.emd_dual_aff(chan_emd2d)
            chan_tree = flex_tree_build.flex_tree_diffusion(chan_aff, 
                                            params.chan_tree_constant)
            
        elif params.chan_affinity_type == DUAL_GAUSSIAN:
            print("Gaussian dual affinity not supported at the moment.")
            return None
        
        message = "Iteration {}: calculating column tree...".format(i)
        
        # constructing channel tree
        if params.chan_tree_type == TREE_TYPE_BINARY:
            chan_tree = bin_tree_build.bin_tree_build(chan_aff,'r_dyadic',
                                                     params.chan_tree_bal_constant)
        elif params.chan_tree_type == TREE_TYPE_FLEXIBLE:
            chan_tree = flex_tree_build.flex_tree_diffusion(chan_aff,
                                                           params.chan_tree_constant)
        dual_chan_trees.append(chan_tree)
        chan_tree_descs.append("Iteration {}".format(i))
        
        # channel tree finished, now starting with rows
        message = "Iteration {}: calculating row affinity...".format(i)

        # calculate row affinity based on column and channel trees
        if params.row_affinity_type == DUAL_EMD:
            row_emd2d = dual_affinity.calc_2demd(np.transpose(data3d, (1, 2, 0)), 
                            dual_col_trees[-1], dual_chan_trees[-1], 
                            row_alpha=params.col_alpha, row_beta=params.col_beta,
                            col_alpha=params.chan_alpha, col_beta=params.chan_beta)
            row_aff = dual_affinity.emd_dual_aff(row_emd2d)
        elif params.row_affinity_type == DUAL_GAUSSIAN:
            print("Gaussian dual affinity not supported at the moment.")
            return None
 
        message = "Iteration {}: calculating row tree...".format(i)

        # constructing row tree
        if params.row_tree_type == TREE_TYPE_BINARY:
            row_tree = bin_tree_build.bin_tree_build(row_aff,'r_dyadic',
                                                     params.row_tree_bal_constant)
        elif params.row_tree_type == TREE_TYPE_FLEXIBLE:
            row_tree = flex_tree_build.flex_tree_diffusion(row_aff,
                                                           params.row_tree_constant)
        dual_row_trees.append(row_tree)
        row_tree_descs.append("Iteration {}".format(i))
        quest_run_desc = "{}".format(datetime.datetime.now())
        
        # calculate column affinity based on row and channel trees
        if params.col_affinity_type == DUAL_EMD:
            col_emd2d = dual_affinity.calc_2demd(np.transpose(data3d, (0, 2, 1)), 
                            dual_row_trees[-1], dual_chan_trees[-1], 
                            row_alpha=params.row_alpha, row_beta=params.row_beta,
                            col_alpha=params.chan_alpha, col_beta=params.chan_beta)
            col_aff = dual_affinity.emd_dual_aff(col_emd2d)
        elif params.col_affinity_type == DUAL_GAUSSIAN:
            print("Gaussian dual affinity not supported at the moment.")
            return None
 
        message = "Iteration {}: calculating column tree...".format(i)
        
        # constructing column tree
        if params.col_tree_type == TREE_TYPE_BINARY:
            col_tree = bin_tree_build.bin_tree_build(col_aff,'r_dyadic',
                                                     params.col_tree_bal_constant)
        elif params.col_tree_type == TREE_TYPE_FLEXIBLE:
            col_tree = flex_tree_build.flex_tree_diffusion(col_aff,
                                                           params.col_tree_constant)
        dual_col_trees.append(col_tree)
        col_tree_descs.append("Iteration {}".format(i))
        quest_run_desc = "{}".format(datetime.datetime.now())
    
        
	# iterations have finished, outputting structures of the tree, 
    # parameters 

    return PyQuest3DRun(quest_run_desc,dual_row_trees,dual_col_trees,dual_chan_trees,
                      row_tree_descs,col_tree_descs,chan_tree_descs,params,
                      init_col_aff,init_row_aff,row_aff,col_aff,chan_aff)
