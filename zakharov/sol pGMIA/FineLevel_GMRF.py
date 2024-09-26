import time
import numpy as np
import statistics
# package for LHS initial sampling
from smt.sampling_methods import LHS
# package for plotting
import matplotlib.pyplot as plt
# self-defined functions
from prec_mat_constr import *
from likelihood_fun import *
from simulation_zakharov import *
from zakharov import *
from CEI import *

def FineLevel_GMRF(comm, this_reg, n_rep, n_init_sol, theta_sol, grid_dim_reg, grid_dim_sol, \
    shift_vector, obj_func, sol_rep, sol_output, sol_var, max_iter, master_generator, size):

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
        # sorted_cei_sol_index is a tuple, sorted_cei_val will not be used
        # find the k region-layer node with largest CEI value (here k is determined by the number of available workers)
        # sort the cei values from largest to smallest and get the corresponding index
        cei_with_index = [(cei_val[i], i) for i in range(n_sol)]
        cei_with_index.sort(reverse = True)
        # the first element in sorted_cei_sol_index is the solution with largest cei value
        sorted_cei_val, sorted_cei_sol_index = zip(*cei_with_index) 

        # add margnal CEI solutions to simulate sequentially until there are k-1 solutions to test
        sol_to_simulate = set([opt_sol])
        cei_start = 0
        while len(sol_to_simulate) < size - 1:
            sol_to_simulate.add(sorted_cei_sol_index[cei_start])
            cei_start += 1
        
        # change the set into list
        sol_to_simulate = list(sol_to_simulate)

        ## simulate the selected solutions, they are always different
        # send data to workers
        for i in range(size - 1):
            sol = sol_to_simulate[i]

            # print('master send to worker in reg', this_reg, 'of sol', sol)

            data = [sol, sampled_sol_index, sol_rep[sol], sol_output[sol], sol_var[sol], obj_func, \
                reg_vec, dim, grid_dim_sol, n_rep, shift_vector, master_generator]
            
            req = comm.isend(data, dest = i + 1, tag = i)
            req.wait()
            # print('send data to worker', i + 1, ' at time:', time.time())

        # receive data from workers
        for i in range(size - 1):
            sol = sol_to_simulate[i]
            req = comm.irecv(source = i + 1, tag = i)
            # print('receive data from worker', i + 1)

            output = req.wait()

            # update simulation output
            [cur_sol_rep, cur_sol_output, cur_sol_var, sol_temp] = output
            sol_rep[sol] = cur_sol_rep
            sol_output[sol] = cur_sol_output
            sol_var[sol] = cur_sol_var
        
        # print('Solution layer finished')

    # return the updated dataset to in the solution layer
    return [sol_rep, sol_output, sol_var, this_reg]


    


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