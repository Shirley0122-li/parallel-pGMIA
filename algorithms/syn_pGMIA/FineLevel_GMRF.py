import numpy as np
import statistics
# package for LHS initial sampling
from smt.sampling_methods import LHS
# package for plotting
# import matplotlib.pyplot as plt
# self-defined functions
from prec_mat_constr import *
from likelihood_fun import *
from simulation_zakharov import *
from zakharov import *
from CEI import *

def FineLevel_GMRF(this_reg, n_rep, n_init_sol, theta_sol, grid_dim_reg, grid_dim_sol, \
    shift_vector, obj_func, sol_rep, sol_output, sol_var, max_iter, master_generator):

    """
    This function runs the solution-layer GMIA:
    if this region is no simulated, initilize the design points to simulate
    find opt_sol and max_cei_sol to simulate

    Input: 
    this_reg (scalar): the index of the selected region-layer node for solution-layer exploration
    n_rep (scalar): number of replications to simulate at each selected solution
    n_init_sol (scalar): number of initial solutions in each region for solution-layer exploration
    theta_sol (1d array, size = dim_sol + 1): solution-layer GMRF parameter
    grid_dim_reg (1d array, size = dim_reg): grid_dim in the region layer
    grid_dim_sol (1d array, size = dim_sol): grid_dim in the solution layer 
    shift_vector (1d vector, size = dim): the first 'dim_sol' are dimensions in the solution layer, and the rest are in the region layer
    obj_func (function): simulate function
    sol_rep (1d array, size = n_sol): number of replications at solutions in this region
    sol_output (1d array, size = n_sol): simulation outputs at solutions in this region
    sol_var (1d array, size = n_sol): simulation output variances at solutions in this region

    Output:
    sol_rep (1d array, size = n_sol) 
    sol_output (1d array, size = n_sol)
    sol_var (1d array, size = n_sol)
    cur_opt_true_val (1d array, size = max_iter)
    cur_max_CEI (1d array, size = max_iter)

    Example to test:
    this_reg = 15
    n_rep = 10
    n_init_sol = 5 
    theta_sol = [0.001, 0.1, 0.1, 0.1]
    grid_dim_reg = np.array([11, 11, 11])
    grid_dim_sol = np.array([11, 11, 11])
    n_sol = np.prod(grid_dim_sol)
    shift_vector = [0,3,2,4,1,5]
    obj_func = simulation_branin
    sol_rep = np.zeros(n_sol)
    sol_output = np.zeros(n_sol)
    sol_var = np.zeros(n_sol)
    FineLevel_GMRF(this_reg, n_rep, n_init_sol, theta_sol, grid_dim_reg, grid_dim_sol, shift_vector, obj_func, sol_rep, sol_output, sol_var)
    """
    
    ## make sure this region has at least n_init_sol sampled solutions for variance computing
    # n_init_sol = 10   # number of initial design soltuions in solution layer
    cur_opt_true_val = np.empty(max_iter)    # record the current optimal value
    cur_max_CEI = np.empty(max_iter)
    reg_vec = np.unravel_index(this_reg, grid_dim_reg)     # tuple of reg_dim integers

    ## compute the solution-layer precision matrix
    n_sol = int(np.prod(grid_dim_sol))   # number of feasible solutions in each region
    dim_sol = len(grid_dim_sol)     # number of dimensions in solution layer
    dim_reg = len(grid_dim_reg)     # number of dimensions in region layer
    dim = dim_sol + dim_reg        # number of total dimensions
    prec_mat = prec_mat_constr(theta_sol, grid_dim_sol)   # precision matrix

    ## check if there are at least two solutions sampled
    sampled_sol_index = np.argwhere(sol_rep != 0)    # n2 * 1 array
    sampled_sol_index = sampled_sol_index.flatten()      # convert 2d array to 1d array
    n2 = len(sampled_sol_index)      # get n2

    # sample new solutions if there is less than 'n_init_sol' sampled solutions
    if n2 < n_init_sol:
            
        # initial design solutions from Latin hypercube sampling
        # Set up solution limits
        sol_bounds = np.array([[0, grid_dim_sol[i]] for i in range(dim_sol)])
        
        # Initialize Latin Hypercube Sampling objects
        sol_sampling = LHS(xlimits=sol_bounds)

        # initial design solutions by latin hypercube sampling
        # # initial design solutions by latin hypercube sampling 
        design_sol = sol_sampling(n_init_sol)   # sample n_initi_sol solutions
        design_sol = np.floor(design_sol)     # n_init_sol * dim_sol ndarray, each row represnets a solution, now 0->10
        design_sol = np.array(design_sol, dtype = np.int64) 

        # sample each design solution if it's not sampled before
        for j in range(n_init_sol):   
            
            # convert the sub vector solution to index
            design_sol_index = np.ravel_multi_index(design_sol[j], grid_dim_sol)

            if design_sol_index not in set(sampled_sol_index):

                # get the original solution for simulation
                design_sol_total = np.concatenate((design_sol[j], np.asarray(reg_vec))) # concatencate the region and solution layers
                orig_design_sol = np.zeros([1, dim], dtype = np.int32)   
                orig_design_sol[:, shift_vector] = design_sol_total    # use the shift vector to get the original design solution, now [0,10]

                # simulate the solution
                sim_output = obj_func(orig_design_sol, n_rep, master_generator)   # [1, n_rep] ndarray, 
                
                # record the simulation output
                sol_rep[design_sol_index] = n_rep
                sol_output[design_sol_index] = statistics.mean(sim_output[0])
                sol_var[design_sol_index] = statistics.variance(sim_output[0])/n_rep

    ## compute the solution-layer beta (GMRF mean)
    sampled_sol_index = np.argwhere(sol_rep != 0)    # n2 * 1 array
    sampled_sol_index = sampled_sol_index.flatten()      # convert 2d array to 1d array
    n2 = len(sampled_sol_index)      # get n2
    beta = likelihood_fun(theta_sol, grid_dim_sol, sampled_sol_index, sol_output[sampled_sol_index], np.diag(sol_var[sampled_sol_index]), 0, 0)[1]

    # print('this_region', this_reg, 'n2', n2)
    # print('sol_index', sampled_sol_index)
    # print('output', sol_output[sampled_sol_index])

    for iter in range(max_iter):

        ## find the optimal solution
        opt_val = np.min(sol_output[np.nonzero(sol_output)])     # current optimal value
        opt_sol = np.where(sol_output == opt_val)[0]     # 1d array
        opt_sol = np.take(opt_sol, 0)   # scalar

        ## find the max cei solution
        # initialize the solution-layer precision matrix Q and intrinsic precision matrix Q_e
        intrinsic_prec_diag = np.zeros(n_sol)
        intrinsic_prec_diag[sampled_sol_index] = 1/sol_var[sampled_sol_index]
        Q_epsi = np.diag(intrinsic_prec_diag)
        Q = prec_mat + Q_epsi
        Q_inv = np.linalg.inv(Q)

        # conditional means
        cum_change = np.dot(Q_epsi, sol_output - beta)
        cond_mean = beta + np.dot(Q_inv[:, sampled_sol_index], cum_change[sampled_sol_index])
        
        # conditional variance and covariance
        cond_var = np.diag(Q_inv)    # 1d array of size(n)
        cond_cov = Q_inv[:, opt_sol].flatten()    # 1d array of size(n)

        # compute solution-layer CEI
        cei_val = CEI(opt_sol, cond_mean, cond_var, cond_cov)
        # find the solution-layer node with largest CEI value
        max_CEI_val = np.max(cei_val)
        cur_max_CEI[iter] = max_CEI_val
        cei_sol = np.where(cei_val == max_CEI_val)[0]      # 1d array
        cei_sol = np.take(cei_sol, 0)      # scalar

        ## simulate the two solutions, they are always different
        for sol in (opt_sol, cei_sol):
            
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

            # record the current optimal true value
            if sol == opt_sol:
                # print(orig_design_sol)
                cur_opt_true_val[iter] = zakharov(orig_design_sol[0] - 5)

            # record the simulation output
            # if this solution has been sampled before
            if sol in set(sampled_sol_index):
                
                total_rep = sol_rep[sol] + n_rep
                # total_mean = (old_mean * old_rep + new_mean * new_rep) / total_rep
                total_mean = (sol_output[sol] * sol_rep[sol] + new_sample_mean * n_rep) / total_rep
                # total_var = 1 / (total_rep - 1) * [1 / total_rep * (old_sum_of_square + new_sum_of_square) - total_mean^2]
                # old_sum_of_square = [old_var * (old_rep - 1) + old_mean^2] * old_rep
                # new_sum_of_square = [new_var * (new_rep - 1) + new_mean^2] * new_rep
                old_sum_of_square = (sol_var[sol] * (sol_rep[sol] - 1) + sol_output[sol] ** 2) * sol_rep[sol]
                new_sum_of_square = (new_sample_var * (n_rep - 1) + new_sample_mean ** 2) * n_rep
                total_var = ((old_sum_of_square + new_sum_of_square) / total_rep  - total_mean ** 2) / (total_rep - 1)

                # record replication, output, and variance
                sol_rep[sol] = total_rep
                sol_output[sol] = total_mean
                sol_var[sol] = total_var
            
            else:    # the solution has not been sampled before 
                
                # record the simulation output
                sol_rep[sol] = n_rep
                sol_output[sol] = statistics.mean(sim_output[0])
                sol_var[sol] = statistics.variance(sim_output[0])/n_rep

    # return the updated dataset to in the solution layer
    return [sol_rep, sol_output, sol_var, cur_opt_true_val, cur_max_CEI]


    


# this_reg = 15
# n_rep = 10
# n_init_sol = 5 
# theta_sol = [0.1, 0.1, 0.1, 0.1]
# grid_dim_reg = np.array([11, 11, 11])
# grid_dim_sol = np.array([11, 11, 11])
# n_sol = np.prod(grid_dim_sol)
# shift_vector = [0,3,2,4,1,5]
# obj_func = simulation_branin
# sol_rep = np.zeros(n_sol)
# sol_output = np.zeros(n_sol)
# sol_var = np.zeros(n_sol)
# FineLevel_GMRF(this_reg, n_rep, n_init_sol, theta_sol, grid_dim_reg, grid_dim_sol, shift_vector, obj_func, sol_rep, sol_output, sol_var)