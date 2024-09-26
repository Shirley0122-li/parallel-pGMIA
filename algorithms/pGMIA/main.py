## import packages
import random
import os
import numpy as np
import math
import statistics
import time
import scipy.optimize   # package for finding GMRF hyperparameter
# package for finding potential projection
from itertools import combinations
# package for sparse linear algebra
from scipy import sparse
# package for LHS initial sampling
from smt.sampling_methods import LHS
# package for plotting
import matplotlib.pyplot as plt
# package for parallelizing
from mpi4py import MPI
import sys

## import self-defined functions
# simulation function
from simulation_branin import *
from simulation_zakharov import *
from zakharov import *
from likelihood_fun import *
from prec_mat_constr import *
from CEI import *
from FineLevel_GMRF import *
from projection_selection import *
import pickle

def main(NAI, max_iter):

# This is the main function for pGMIA
# Input:
# NAI (scalar): random seed
# max_iter (scalar): simulation budget 

# DOES NOT HAVE OUTPUT
# The result is saved in data

    # random.seed(NAI)
    # os.environ['PYTHONHASHSEED'] = str(NAI)

    ########## Initialization ###############################################################################

    ## problem and pGMIA initial setting
    obj_func = simulation_zakharov
    side = 11   # number of feasible solutions in each dimension
    dim = 6     # number of dimensions
    # vector of the no. of feasible solutions in each dimension: [11 11 11 11 11 11]' 
    grid_dim = side * np.ones(dim, dtype = np.int32) 
    n_all = int(np.product(grid_dim))    # number of all feasible solutions
    n_rep = 10    # number of replications when visiting a solution
    n_init_reg = 5  # number of initial design regions
    n_init_sol = 5  # number of initial design solutions within each design region
    period = 100    # period for pGMIA to update the projection

    ## projection initilization
    dim_reg = 3   # number of region-layer dimensions
    dim_sol = dim - dim_reg   # number of solution-layer dimensions
    
    # a list of combinations for dimensions in region layer (each is a tuple)
    combines = list(combinations(range(dim), dim_reg))
    n_proj = len(combines)    # number of projections

    # random select a projection for initialization
    choice = random.randint(0, n_proj-1)

    # construct shift vector, the first 'dim_sol' are the dimensions in the solution layer, and the rest are the dimensons in the region layer
    temp = set(combines[choice])
    shift_vector = []
    for i in range(dim):
        if i not in temp:
            shift_vector.append(i)
    
    for i in temp:
        shift_vector.append(i)
    
    # initialization of projection
    grid_dim_shift = grid_dim[shift_vector]  # shifted grid_dimension
    grid_dim_reg = grid_dim_shift[dim_reg:dim]  # grid_dim in the region layer
    grid_dim_sol = grid_dim_shift[0:dim_reg]   # grid_dim in the solution layer
    n_reg = int(np.prod(grid_dim_reg))   # number of feasible regions in the region layer
    n_sol = int(np.prod(grid_dim_sol))   # number of feasible solutions in each region

    ## dataset initialization

    # single-layer -> all solutions
    # [i,j] of the following matrices represents the jth solution at ith layer
    # NOTICE: np.empty does not set the array values to zero, and may therefore by marginally faster, should be used with caution!!
    all_sol_rep = np.zeros([n_reg, n_sol])    # number of replications at each solution
    all_sol_output = np.zeros([n_reg, n_sol])   # output mean at each solution
    all_sol_var = np.zeros([n_reg, n_sol])    # output variance at each solution

    # region-layer -> all region-layer nodes
    # the [i] of the following vector represents the ith region
    all_reg_output = np.zeros(n_reg)  # output mean at each region
    all_reg_var = np.zeros(n_reg)   # output variance at each region

    # each iteration

    # region layer
    cur_opt_reg = np.empty(max_iter)     # current optimal region (with smallest region output)
    cur_opt_sol_reg = np.empty(max_iter)   # region contains the current optimal solution
    cur_max_CEI_reg = np.empty(max_iter)   # region that has the largest CEI value
    cur_reg_CEI_max_val = np.empty(max_iter)   # the largest region-layer CEI value
    
    # solution layer
    cur_opt_sol = np.empty([max_iter, dim])  # current optimal solution 
    cur_opt_val = np.empty(max_iter)     # current optimal value
    cur_opt_true_val = np.empty([max_iter, 3])    # current true value at optimal solution
    cur_num_sampled = np.empty(max_iter)    # number of sampled solutions
    cur_num_reps = np.empty(max_iter)      # number of total replications

    # GMRF hyperparameters
    n_period = max_iter // period   # number of period
    cur_shift_vector = np.empty([n_period, dim])  # shift vector at each period
    cur_beta = np.empty(n_period)     # current GMRF mean hyperparameter
    cur_theta = np.empty([n_period, dim + 1])   # current single-layer hyperparameters
    cur_tau = np.empty([n_period, dim_reg + 1])    # current region-layer hyperparameters

    ## simulate the initial design regions and design solutions
    # initial design regions by latin hypercube sampling
    reg_limits = [[0, grid_dim_reg[i]] for i in range(dim_reg)]
    reg_limits = np.array(reg_limits)
    reg_sampling = LHS(xlimits = reg_limits, criterion = 'maximin')

    sol_limits = [[0, grid_dim_sol[i]] for i in range(dim_sol)]
    sol_limits = np.array(sol_limits)
    sol_sampling = LHS(xlimits = sol_limits, criterion = 'maximin')

    design_reg = reg_sampling(n_init_reg)   # sample n_init_reg regions
    design_reg = np.floor(design_reg)   # n_init_reg * dim_reg ndarry, each row represents a region, now 0->10
    design_reg = np.array(design_reg, dtype = np.int64)
    
    # convert the sub vector of region to index 
    design_reg_index = np.empty(n_init_reg, dtype = np.int64)
    for i in range(n_init_reg):
        # sub2ind
        design_reg_index[i] = np.ravel_multi_index(design_reg[i], grid_dim_reg)
        # if want ind2sub
        # temp = np.unravel_index(design_reg_index[i], grid_dim_reg)
        
        # initial design solutions by latin hypercube sampling 
        design_sol = sol_sampling(n_init_sol)   # sample n_initi_sol solutions
        design_sol = np.floor(design_sol)     # n_init_sol * dim_sol ndarray, each row represnets a solution, now 0->10
        design_sol = np.array(design_sol, dtype = np.int64)

        for j in range(n_init_sol):
            # convert the sub vector solution to index
            design_sol_index = np.ravel_multi_index(design_sol[j], grid_dim_sol)

            # get the original solution for simulation
            design_sol_total = np.concatenate((design_sol[j], design_reg[i])) # concatencate the region and solution layers
            orig_design_sol = np.zeros([1, dim], dtype = np.int32)   
            orig_design_sol[:,shift_vector] = design_sol_total    # use the shift vector to get the original design solution, now [0,10]

            # simulate the solution
            sim_output = obj_func(orig_design_sol, n_rep)   # [1, n_rep] ndarray, 
            
            # record the simulation output
            all_sol_rep[design_reg_index[i], design_sol_index] = n_rep
            all_sol_output[design_reg_index[i], design_sol_index] = statistics.mean(sim_output[0])
            all_sol_var[design_reg_index[i], design_sol_index] = statistics.variance(sim_output[0])/n_rep
            # print('reg', design_reg_index[i], 'sol', design_sol_index, 'output', all_sol_output[design_reg_index[i], design_sol_index], 'var', all_sol_var[design_reg_index[i], design_sol_index])

        # record the region-layer simulation output
        # find the sampled sol in this region 
        sampled_index = np.argwhere(all_sol_rep[design_reg_index[i]] != 0)    # n_init_sol * 1 array
        sampled_index = sampled_index.flatten()   # convert 2d array to 1d array
        sampled_output = all_sol_output[design_reg_index[i], sampled_index]    # n_init_sol array
        all_reg_output[design_reg_index[i]] = statistics.mean(sampled_output)
        all_reg_var[design_reg_index[i]] = statistics.variance(sampled_output) / n_init_sol * (1 - n_init_sol / n_sol) + statistics.mean(sampled_output) / n_init_sol
        # print('reg', design_reg_index[i], 'output', all_reg_output[design_reg_index[i]], 'var', all_reg_var[design_reg_index[i]])

    del reg_limits, design_reg, sampled_index, sampled_output
    
    ## MLE for region and solution layers GMRF hyperparameters
    
    ## MLE for region and solution layers GMRF hyperparameters
    
    [beta, tau, theta_sol] = MLE(all_reg_output, all_reg_var, all_sol_output, all_sol_var, \
        grid_dim_reg, grid_dim_sol, design_reg_index, dim_reg, dim_sol, n_sol)

    theta_sol[0] = 1/theta_sol[0]

    ## record the GMRF hyperparameter
    theta = np.zeros(dim + 1)
    theta[0] = theta_sol[0]
    # get the theta at solution-layer dimensions
    temp_sol_shift = [shift_vector[i] + 1 for i in range(dim_sol)]
    theta[temp_sol_shift] = theta_sol[1:dim_sol+1]
    # compute the theta at region-layer dimensions
    for i in range(dim_reg):
        theta[shift_vector[dim_sol+i] + 1] = tau[0]*tau[i+1]/(n_sol*theta[0])
    # record the GMRF parameters
    # print('shift_vector:', shift_vector)
    cur_shift_vector[0, :] = shift_vector  # shift vector at each period
    cur_beta[0] = beta     # current GMRF mean hyperparameter
    cur_theta[0, :] = theta   # current single-layer hyperparameters
    cur_tau[0, :] = tau    # current region-layer hyperparametes

    ## initialize the region-layer precision matrix T and intrinsic precision matrix T_e
    reg_prec_mat = prec_mat_constr(tau, grid_dim_reg)
    intrinsic_prec_diag = np.zeros(n_reg)
    intrinsic_prec_diag[design_reg_index] = 1/all_reg_var[design_reg_index]
    T_epsi = np.diag(intrinsic_prec_diag)
    # record start time
    start_time = time.time()

    print('Initialization finished.')

    #########################################################################################################
    
    ## start pGMIA with fixed budget -> max_iter
    for iter in range(max_iter):
        
        # print('iter: ', iter)
        
        time_start = time.time()

        # get inverse of region-layer precision matrix
        T = reg_prec_mat + T_epsi
        T_inv = np.linalg.inv(T)

        # find the region-layer optimal node
        opt_val = np.min(all_reg_output[np.nonzero(all_reg_output)])     # current optimal value
        opt_reg = np.where(all_reg_output == opt_val)[0]     # 1 array
        opt_reg = np.take(opt_reg, 0)      # scalar
        cur_opt_reg[iter] = opt_reg       # record

        # find the current optimal solution and the region-layer node that contains the current optimal solution
        opt_val = np.min(all_sol_output[np.nonzero(all_sol_output)])
        opt_sol_reg, opt_s = np.where(all_sol_output == opt_val)
        opt_sol_reg = np.take(opt_sol_reg, 0)      # 1d array -> scalar
        opt_s = np.take(opt_s, 0)        # 1d array -> scalar
        cur_opt_sol_reg[i] = opt_sol_reg   

        # opt solution 
        opt_reg_vec = np.unravel_index(opt_sol_reg, grid_dim_reg)     # tuple of reg_dim integers
        opt_sol_vec = np.unravel_index(opt_s, grid_dim_sol)           # tuple of sol_dim integers

        # get the original solution for recording
        opt_sol_total = np.concatenate((np.asarray(opt_sol_vec), np.asarray(opt_reg_vec))) # concatencate the region and solution layers
        opt_sol_total = opt_sol_total.flatten()   # convert 2d array to 1d array
        orig_opt_sol = np.zeros([1, dim], dtype = np.int32)     # 1 * dim array
        orig_opt_sol[:, shift_vector] = opt_sol_total    # use the shift vector to get the original design solution, now [0,10]
        # record
        cur_opt_sol[iter] = orig_opt_sol[0]   # current optimal solution 
        cur_opt_val[iter] = opt_val     # scalar
        total_num_visited = int(np.sum(all_sol_rep) / 10) 
        # cur_opt_true_val[iter] = zakharov(orig_opt_sol[0] - 5)
        temp_array = np.array([zakharov(orig_opt_sol[0] - 5), time.time()-start_time, total_num_visited])
        # print(temp_array)
        cur_opt_true_val[iter, :] = temp_array.reshape(1,len(temp_array))
        
        # compute CEI
        # find the sampled regions
        sampled_reg_index = np.argwhere(all_reg_output != 0)    # n2 * 1 array
        sampled_reg_index = sampled_reg_index.flatten()    # convert 2d array to 1d array
        
        # conditional means
        cum_change = np.dot(T_epsi, all_reg_output - beta)
        reg_cond_mean = beta + np.dot(T_inv[:, sampled_reg_index], cum_change[sampled_reg_index])
        
        # conditional variance and covariance
        reg_cond_var = np.diag(T_inv)    # 1d array of size(n)
        reg_cond_cov = T_inv[:, opt_reg].flatten()    # 1d array of size(n)

        # compute region-layer CEI
        cei_val = CEI(opt_reg, reg_cond_mean, reg_cond_var, reg_cond_cov)
        # find the region-layer node with largest CEI value
        max_CEI_val = np.max(cei_val)
        cei_reg = np.where(cei_val == max_CEI_val)[0]    # 1d array
        cei_reg = np.take(cei_reg, 0)      # scalar
        # record
        cur_max_CEI_reg[iter] = cei_reg     # region that has the largest CEI value, scalar
        cur_reg_CEI_max_val[iter] = max_CEI_val   # scalar

        # record number of sampled solutions and replications
        cur_num_sampled[iter] = int(np.count_nonzero(all_sol_rep))
        cur_num_reps[iter] = int(np.sum(all_sol_rep))

        # simulate the three selected regions, use 'set' to avoid running the same region
        for reg in set([opt_reg, opt_sol_reg, cei_reg]):
            
            # for each region, run solution-layer GMIA
            output = FineLevel_GMRF(reg, n_rep, n_init_sol, theta_sol, grid_dim_reg, grid_dim_sol, shift_vector, obj_func, \
                all_sol_rep[reg], all_sol_output[reg], all_sol_var[reg])
            
            # update the dataset 
            all_sol_rep[reg], all_sol_output[reg], all_sol_var[reg] = output
            # find the sampled sol in this region 
            sampled_index = np.argwhere(all_sol_rep[reg] != 0)    # n_sampled_sol * 1 array
            sampled_index = sampled_index.flatten()   # convert 2d array to 1d array
            n_sampled_sol = len(sampled_index)
            sampled_output = all_sol_output[reg, sampled_index]    # n_sampled_sol array
            all_reg_output[reg] = statistics.mean(sampled_output)
            all_reg_var[reg] = statistics.variance(sampled_output) / n_sampled_sol * (1 - n_sampled_sol / n_sol) + statistics.mean(sampled_output) / n_sampled_sol

    #########################################################################################################
        
        # update the projection every p iterations
        if iter % period == 0:
            print('iter: ', iter)
            # find all sampled solutions and transfer it into n2 * dim array
            sampled_temp = np.nonzero(all_sol_rep)
            n_sampled = len(sampled_temp[0])     # number of sampled solutions
            reg_vec_temp = np.unravel_index(sampled_temp[0], grid_dim_reg)   # tuple of vectors at region layer 
            sol_vec_temp = np.unravel_index(sampled_temp[1], grid_dim_sol)   # tuple of vectors at solution layer
            shift_sampled_vec = np.transpose(np.concatenate((sol_vec_temp, reg_vec_temp)))    # n2 * dim array of vectors after shifting
            
            # create an empty n2 * dim array
            sampled_vec = np.zeros([n_sampled, dim], dtype = np.int32)
            sampled_vec[:, shift_vector] = shift_sampled_vec     # n2 * dim array, each row represents a sampled solution

            # sampled output and sampled variance
            sampled_rep = all_sol_rep[sampled_temp]          # 1d array, size = n2
            sampled_output = all_sol_output[sampled_temp]      # 1d array, size = n2
            sampled_var = all_sol_var[sampled_temp]        # 1d array, size = n2

            # select a projection and return the parameters and updated dataset
            output = projection_selection(n_init_reg, n_init_sol, n_rep, grid_dim, dim_reg, obj_func, sampled_vec, sampled_rep, sampled_output, sampled_var)

            # update the dataset in the new projection
            [shift_vector, theta, beta, tau, theta_sol, reg_prec_mat, \
                grid_dim_reg, grid_dim_sol, n_reg, n_sol, \
                all_sol_rep, all_sol_output, all_sol_var, all_reg_output, all_reg_var] = output
            
            cur_shift_vector[iter // period, :] = shift_vector  # shift vector at each period
            cur_beta[iter // period] = beta     # current GMRF mean hyperparameter
            cur_theta[iter // period, :] = theta   # current single-layer hyperparameters
            cur_tau[iter // period, :] = tau    # current region-layer hyperparametes


        # update the region-layer precision matrix T and intrinsic precision matrix T_e
        sampled_reg_index = np.argwhere(all_reg_output != 0)    # n2 * 1 array
        sampled_reg_index = sampled_reg_index.flatten()    # convert 2d array to 1d array

        intrinsic_prec_diag = np.zeros(n_reg)
        intrinsic_prec_diag[sampled_reg_index] = 1/all_reg_var[sampled_reg_index]
        T_epsi = np.diag(intrinsic_prec_diag) 
        # print('sampled_reg_index', sampled_reg_index)  

        time_end = time.time()

    return [cur_opt_sol, cur_opt_true_val]


seed = int(sys.argv[1])
output = main(seed,3000)

file_name = 'seed' + str(sys.argv[1]) + 'seq_total'
# open a file, where you ant to store the data
file = open(file_name, 'wb')
# dump information to that file
pickle.dump(output, file)
# close the file
file.close()

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()

# if rank == 0:
#     output = main(1, 2000)
#     plt.plot(output[1])
#     plt.savefig('cur_opt_val.png')
# else: