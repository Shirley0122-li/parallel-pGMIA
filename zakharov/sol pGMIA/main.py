## import packages
import random
import os
import numpy as np
import math
import statistics
import time
import pickle
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
# logging
import sys

## import self-defined functions
# simulation function
from simulation_zakharov import *
from likelihood_fun import *
from likelihood_new import *
from prec_mat_constr import *
from CEI import *
from FineLevel_GMRF import *
from projection_selection import *
from MLE import *

def pGMIA(file_name, NAI, max_iter, size, master_generator):

# This is the main function for pGMIA
# Input:
# NAI (scalar): random seed
# max_iter (scalar): simulation budget 

    random.seed(NAI)
    os.environ['PYTHONHASHSEED'] = str(NAI)

    ########## Initialization ###############################################################################

    ## problem and pGMIA initial setting
    obj_func = simulation_zakharov
    true_obj_func = zakharov
    side = 11   # number of feasible solutions in each dimension
    dim = 6     # number of dimensions
    # vector of the no. of feasible solutions in each dimension: [11 11 11 11 11 11]' 
    grid_dim = side * np.ones(dim, dtype = np.int32) 
    n_all = int(np.product(grid_dim))    # number of all feasible solutions
    n_rep = 10    # number of replications when visiting a solution
    n_init_reg = 5  # number of initial design regions
    n_init_sol = 5  # number of initial design solutions within each design region
    period = 200    # period for pGMIA to update the projection

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
    cur_sampled_in_opt_reg = np.empty(max_iter)    # the number of sampled solutions in the opt region (5,5,5) index = 665
    
    # solution layer
    cur_opt_sol = np.empty([max_iter, dim])  # current optimal solution 
    cur_opt_val = np.empty(max_iter)     # current optimal value
    cur_opt_true_val = np.empty([max_iter, 3])    # current true value at optimal solution
    # cur_opt_true_val = np.empty(max_iter)    # current true value at optimal solution
    cur_num_sampled = np.empty(max_iter)    # number of sampled solutions
    cur_num_reps = np.empty(max_iter)      # number of total replications
    cur_run_time = np.empty(max_iter)      # running time 
    cur_total_opt_val = np.array([])       # !!! to check if it's same as cur_opt_true_val
    cur_total_max_CEI = np.array([])       # record the CEI value at iterations in the solution layer
    
    # records in each visited region
    cur_max_CEI_in_reg = np.array([])      # the max CEI value in each visited region
    cur_sol_opt_in_reg = np.array([])      # the true objectivee function val in each visted region
    cur_num_sampled_in_reg = np.array([])     # the number of sampled solutions in each visited region
    cur_num_reps_in_reg = np.array([])       # the number of replications ran in each visited region


    # GMRF hyperparameters
    n_period = max_iter // period   # number of period
    cur_shift_vector = np.empty([n_period, dim])  # shift vector at each period
    cur_beta = np.empty(n_period)     # current GMRF mean hyperparameter
    cur_theta = np.empty([n_period, dim + 1])   # current single-layer hyperparameters
    cur_tau = np.empty([n_period, dim_reg + 1])    # current region-layer hyperparameters

    ## simulate the initial design regions and design solutions
    # initial design regions by latin hypercube sampling
    # Set up region and solution limits
    reg_bounds = np.array([[0, grid_dim_reg[i]] for i in range(dim_reg)])
    sol_bounds = np.array([[0, grid_dim_sol[i]] for i in range(dim_sol)])
    
    # Initialize Latin Hypercube Sampling objects
    reg_sampling = LHS(xlimits=reg_bounds)
    sol_sampling = LHS(xlimits=sol_bounds)

    # Sample n_init_reg regions
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
            sim_output = obj_func(orig_design_sol, n_rep, master_generator)   # [1, n_rep] ndarray, 
            
            # record the simulation output
            all_sol_rep[design_reg_index[i], design_sol_index] = n_rep
            all_sol_output[design_reg_index[i], design_sol_index] = statistics.mean(sim_output[0])
            all_sol_var[design_reg_index[i], design_sol_index] = statistics.variance(sim_output[0])/n_rep

        # record the region-layer simulation output
        # find the sampled sol in this region 
        sampled_index = np.argwhere(all_sol_rep[design_reg_index[i]] != 0)    # n_init_sol * 1 array
        sampled_index = sampled_index.flatten()   # convert 2d array to 1d array
        sampled_output = all_sol_output[design_reg_index[i], sampled_index]    # n_init_sol array
        all_reg_output[design_reg_index[i]] = statistics.mean(sampled_output)
        all_reg_var[design_reg_index[i]] = statistics.variance(sampled_output) / n_init_sol * (1 - n_init_sol / n_sol) + statistics.mean(sampled_output) / n_init_sol
    
    del design_reg, sampled_index, sampled_output
    
    # output = [all_reg_output, all_reg_var, design_reg_index, all_sol_output, all_sol_var, \
    #     grid_dim_reg, grid_dim_sol, design_reg_index, dim_reg, dim_sol, n_sol]
    # file = open('initial_output', 'wb')
    # # dump information to that file
    # pickle.dump(output, file)
    # # close the file
    # file.close()

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
        
        iter_start = time.time()
        # print('iter: ', iter, ' at time ', iter_start - start_time)

        # get inverse of region-layer precision matrix
        T = reg_prec_mat + T_epsi
        T_inv = np.linalg.inv(T)

        # find the region-layer optimal node
        opt_val = np.min(all_reg_output[np.nonzero(all_reg_output)])     # current optimal value
        # print('sampled_reg_output', all_reg_output[np.nonzero(all_reg_output)])
        # print('region layer opt value', opt_val)
        opt_reg = np.where(all_reg_output == opt_val)[0]     # 1 array
        opt_reg = np.take(opt_reg, 0)      # scalar
        cur_opt_reg[iter] = opt_reg       # record

        # find the current optimal solution and the region-layer node that contains the current optimal solution
        opt_val = np.min(all_sol_output[np.nonzero(all_sol_output)])
        opt_sol_reg, opt_s = np.where(all_sol_output == opt_val)
        opt_sol_reg = np.take(opt_sol_reg, 0)      # 1d array -> scalar
        opt_s = np.take(opt_s, 0)        # 1d array -> scalar
        cur_opt_sol_reg[iter] = opt_sol_reg   

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
        # cur_opt_true_val[iter] = true_obj_func(orig_opt_sol[0] - 5)
        temp_array = np.array([true_obj_func(orig_opt_sol[0] - 5), time.time()-start_time, total_num_visited])
        # print(temp_array)
        cur_opt_true_val[iter, :] = temp_array.reshape(1,len(temp_array))
        # print('current optimal solution', orig_opt_sol[0])
        # print('cur_opt_true', cur_opt_true_val[iter])

        # record the number of sampled solutions in true optimal region
        temp = np.argwhere(all_sol_output[665] != 0)    # n2 * 1 array
        temp = temp.flatten()
        cur_sampled_in_opt_reg[iter] = len(temp)
        
        # compute CEI
        # find the sampled regions
        sampled_reg_index = np.argwhere(all_reg_output != 0)    # n2 * 1 array
        sampled_reg_index = sampled_reg_index.flatten()    # convert 2d array to 1d array
        # print('sampled_reg_index', sampled_reg_index)
        
        # conditional means
        cum_change = np.dot(T_epsi, all_reg_output - beta)
        reg_cond_mean = beta + np.dot(T_inv[:, sampled_reg_index], cum_change[sampled_reg_index])
        
        # conditional variance and covariance
        reg_cond_var = np.diag(T_inv)    # 1d array of size(n)
        reg_cond_cov = T_inv[:, opt_reg].flatten()    # 1d array of size(n)

        # compute region-layer CEI
        cei_val = CEI(opt_reg, reg_cond_mean, reg_cond_var, reg_cond_cov)
        # find the k region-layer node with largest CEI value (here k is determined by the number of available workers)
        # sort the cei values from largest to smallest and get the corresponding index
        cei_with_index = [(cei_val[i], i) for i in range(n_reg)]
        cei_with_index.sort(reverse = True)
        # the first element in sorted_cei_reg_index is the region with largest cei value
        # sorted_cei_reg_index is a tuple, sorted_cei_val will not be used
        sorted_cei_val, sorted_cei_reg_index = zip(*cei_with_index)    

        # record
        cei_reg = sorted_cei_reg_index[0]
        cur_max_CEI_reg[iter] = sorted_cei_reg_index[0]     # region that has the largest CEI value, scalar
        cur_reg_CEI_max_val[iter] = sorted_cei_val[0]   # scalar

        # record number of sampled solutions and replications
        cur_num_sampled[iter] = int(np.count_nonzero(all_sol_rep))
        cur_num_reps[iter] = int(np.sum(all_sol_rep))

        # simulate the three selected regions, use 'set' to avoid running the same region
        # for reg in set([opt_reg, opt_sol_reg, cei_reg]):
        reg_to_simulate = [opt_reg, opt_sol_reg, cei_reg]
        # reg_to_simulate = list(reg_to_simulate)

        # send data to workers
        for reg in reg_to_simulate:

            sol_rep = all_sol_rep[reg]
            sampled_sol_index = np.argwhere(sol_rep != 0)    # n2 * 1 array
            sampled_sol_index = sampled_sol_index.flatten()      # convert 2d array to 1d array
            
            output = FineLevel_GMRF(comm, reg, n_rep, n_init_sol, theta_sol, grid_dim_reg, grid_dim_sol, shift_vector, obj_func, \
                all_sol_rep[reg], all_sol_output[reg], all_sol_var[reg], 1, processor_generator, size)

            # update the dataset 
            all_sol_rep[reg], all_sol_output[reg], all_sol_var[reg], reg = output
            # find the sampled sol in this region 
            sampled_index = np.argwhere(all_sol_rep[reg] != 0)    # n_sampled_sol * 1 array
            sampled_index = sampled_index.flatten()   # convert 2d array to 1d array
            # print('region', reg, 'sampled_index', sampled_index)
            n_sampled_sol = len(sampled_index)
            sampled_output = all_sol_output[reg, sampled_index]    # n_sampled_sol array
            all_reg_output[reg] = statistics.mean(sampled_output)
            all_reg_var[reg] = statistics.variance(sampled_output) / n_sampled_sol * (1 - n_sampled_sol / n_sol) + statistics.mean(sampled_output) / n_sampled_sol

    #########################################################################################################
        
        # update the projection every p iterations
        if iter % period == period - 1:
            
            # print('iter: ', iter, ' at time ', iter_start - start_time)
            # # open a file, where you ant to store the data
            # file = open(file_name, 'wb')
            # # dump information to that file
            # output = [cur_opt_sol, cur_opt_true_val, cur_opt_val, cur_num_sampled, cur_num_reps, \
            #     cur_opt_reg, cur_opt_sol_reg, cur_max_CEI_reg, cur_reg_CEI_max_val, cur_total_opt_val, \
            #     cur_sampled_in_opt_reg, cur_total_max_CEI, cur_run_time]
            # pickle.dump(output, file)
            # # close the file
            # file.close()

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
            output = projection_selection(n_init_reg, n_init_sol, n_rep, grid_dim, dim_reg, obj_func, \
                sampled_vec, sampled_rep, sampled_output, sampled_var, master_generator)

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

        iter_end = time.time()
        cur_run_time[iter] = iter_end - iter_start
        
        


    return [cur_opt_sol, cur_opt_true_val, cur_opt_val, cur_num_sampled, cur_num_reps, \
        cur_opt_reg, cur_opt_sol_reg, cur_max_CEI_reg, cur_reg_CEI_max_val, cur_total_opt_val, \
            cur_sampled_in_opt_reg, cur_total_max_CEI, cur_run_time]


# output = main(1, 2000)
# plt.plot(output[1])
# plt.savefig('cur_opt_val.png')

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# print(sys.argv)
seed = int(sys.argv[1])
# print(sys.argv[2])
seed_on_this_processor = size * seed + rank
bit_generator = np.random.MT19937(seed_on_this_processor)
processor_generator = np.random.Generator(bit_generator)
max_iter = 1000

if rank == 0:
    print('start the pGMIA')

    opt_true_val = []
    num_sampled = []
    num_reps = []
    opt_reg = []
    opt_sol_reg = []
    max_CEI_reg = []
    total_opt_val = []
    run_time = []
    file_name = 'seed' + str(sys.argv[1]) + 'total'

    output = pGMIA(file_name, seed, max_iter, size, processor_generator)

    # opt_true_val.append(output[1])
    # num_sampled.append(output[3])
    # num_reps.append(output[4])
    # opt_reg.append(output[5])
    # opt_sol_reg.append(output[6])
    # max_CEI_reg.append(output[7])
    # total_opt_val.append(output[9])
    # run_time.append(output[10])


    # open a file, where you ant to store the data
    file = open(file_name, 'wb')
    # dump information to that file
    pickle.dump(output, file)
    # close the file
    file.close()
        
    
    # plt.plot(output[1])
    # plt.savefig('cur_opt_val.png')

else:

    for iter in range(3 * max_iter):
        # receive data from the master
        i = rank - 1
        worker_start = time.time()
        req = comm.irecv(source=0, tag=i)
        data = req.wait()
        # worker_end = time.time()
        # print('worker', rank, 'receive data from master')

        # record the data from the master
        [sol, sampled_sol_index, cur_sol_rep, cur_sol_output, cur_sol_var, obj_func, \
                reg_vec, dim, grid_dim_sol, n_rep, shift_vector, master_generator] = data
        
        # for each solution, parallel simulation
        # convert the solution index to sub vector
        sol_vec = np.unravel_index(sol, grid_dim_sol)     # tuple of sol_dim integers

        # get the original solution for simulation
        design_sol_total = np.concatenate((np.asarray(sol_vec), np.asarray(reg_vec))) # concatencate the region and solution layers
        orig_design_sol = np.zeros([1, dim], dtype = np.int32)   
        orig_design_sol[:, shift_vector] = design_sol_total    # use the shift vector to get the original design solution, now [0,10]

        # simulate the solution
        sim_output = obj_func(orig_design_sol, n_rep, master_generator)   # [1, n_rep] ndarray, 
        new_sample_mean = statistics.mean(sim_output[0])
        new_sample_var = statistics.variance(sim_output[0])/n_rep

        # record the simulation output
        # if this solution has been sampled before
        if sol in set(sampled_sol_index):
            
            total_rep = cur_sol_rep + n_rep
            # total_mean = (old_mean * old_rep + new_mean * new_rep) / total_rep
            total_mean = (cur_sol_output * cur_sol_rep + new_sample_mean * n_rep) / total_rep
            # total_var = 1 / (total_rep - 1) * [1 / total_rep * (old_sum_of_square + new_sum_of_square) - total_mean^2]
            # old_sum_of_square = [old_var * (old_rep - 1) + old_mean^2] * old_rep
            # new_sum_of_square = [new_var * (new_rep - 1) + new_mean^2] * new_rep
            old_sum_of_square = (cur_sol_var * (cur_sol_rep - 1) + cur_sol_output ** 2) * cur_sol_rep
            new_sum_of_square = (new_sample_var * (n_rep - 1) + new_sample_mean ** 2) * n_rep
            total_var = ((old_sum_of_square + new_sum_of_square) / total_rep  - total_mean ** 2) / (total_rep - 1)

            # record replication, output, and variance
            cur_sol_rep = total_rep
            cur_sol_output = total_mean
            cur_sol_var = total_var
        
        else:    # the solution has not been sampled before 
            
            # record the simulation output
            cur_sol_rep = n_rep
            cur_sol_output = statistics.mean(sim_output[0])
            cur_sol_var = statistics.variance(sim_output[0])/n_rep

        output = [cur_sol_rep, cur_sol_output, cur_sol_var]
        # send data to the master
        # worker_start = time.time()
        output.append(sol)
        req = comm.isend(output, dest=0, tag=i)
        req.wait()
        # print('worker', rank, 'solution layer ', 'time:', worker_end - worker_start)
        # print('worker', rank, 'send data to master', ' at time:', time.time())



    # mpiexec -n 8 python main.py
    # mpiexec -n 8 python -u main.py 1 'seed1'