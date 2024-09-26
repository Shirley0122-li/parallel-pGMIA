import numpy as np
import random
import statistics
import time
import scipy.optimize   # package for finding GMRF hyperparameter

# package for finding potential projection
from itertools import combinations
# package for LHS initial sampling
from smt.sampling_methods import LHS


## import self-defined functions
from likelihood_fun import *
from likelihood_new import *
from simulation_zakharov import *
from prec_mat_constr import *
from MLE import *

def projection_selection(n_init_reg, n_init_sol, n_rep, grid_dim, dim_reg, obj_func, sampled_vec, \
    sampled_rep, sampled_mean, sampled_var):

# This function selects a new projection and then returns the selected shift vector and region-layer GMRF parameters
#
# Input:
# n_init_reg (scalar): number of region-layer nodes to initialize
# n_init_sol (scalar): number of solution-layer nodes to initialize
# n_rep (scalar): number of replications at each sampled solution
# grid_dimensions (1d vector, size = dim): vector of the no. of feasible solutions in each dimension
# obj_func (function): the objective function of the problem
# dim_reg (scalar): number of dimensions in the region layer
# sampled_vec (2d araay, size = n2 * dim): each row is a 'dim' size representing the vector of each sampled point
# sampled_rep (1d array, size = n2): each row is the number of replications sampled at each sampled point
# sampled_mean (1d array, size = n2): each row is the sampled mean at each point
# sampled_var (1d array, size = n2): each row is the sampled variance at each point
# 
# Output:
# shift_vector (1d list, size = dim) 
# theta (1d list, size = dim + 1)
# beta (scalar)
# tau (1d list, size = dim_reg + 1)
# theta_sol (1d list, size = dim_sol + 1)
# reg_prec_mat (2d array, size = n_reg * n_reg)
# grid_dim_reg (1d list, size = dim_reg)
# grid_dim_sol (1d list, size = dim_sol)
# n_reg (scalar)
# n_sol (scalar)
# all_sol_rep (2d array, size = n_reg * n_sol)
# all_sol_output (2d array, size = n_reg * n_sol)
# all_sol_var (2d array, size = n_reg * n_sol)
# all_reg_output (1d array, size = n_reg)
# all_reg_var (1d array, size = n_reg)
# 
# Example to test:
# side = 11   # number of feasible solutions in each dimension
# dim = 6     # number of dimensions
# vector of the no. of feasible solutions in each dimension: [11 11 11 11 11 11]' 
# grid_dim = side * np.ones(dim, dtype = np.int32) 
# dim_reg = 3
# n_init_reg = 5
# n_init_sol = 5
# n_rep = 10
# obj_func = simulation_zakharov

# sampled_vec = np.array([[3,4,5,2,1,7], [3,2,6,5,4,7], [4,7,9,5,1,3], [0,5,5,6,7,2]])
# sampled_rep = np.array([10,10,10,10])
# sampled_mean = np.array([4.454, 3.5435, 76.564, 23.34])
# sampled_var = np.array([0.232, 0.555, 0.343, 0.567])

# projection_selection(n_init_reg, n_init_sol, n_rep, grid_dim, dim_reg, obj_func, sampled_vec, sampled_rep, sampled_mean, sampled_var)
    
    
    # print(sampled_vec)
    # dimensions of the problem
    n_init_sol = 2
    dim = len(grid_dim)
    dim_sol = dim - dim_reg
    # a list of combinations for dimensions in region layer (each is a tuple)
    combines = list(combinations(range(dim), dim_reg))
    n_proj = len(combines)    # number of projections

    # random select a projection as the selection criterion
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
    all_sol_rep = np.zeros([n_reg, n_sol])    # number of replications at each solution
    all_sol_output = np.zeros([n_reg, n_sol])   # output mean at each solution
    all_sol_var = np.zeros([n_reg, n_sol])    # output variance at each solution

    # region-layer -> all region-layer nodes
    # the [i] of the following vector represents the ith region
    all_reg_output = np.zeros(n_reg)  # output mean at each region
    all_reg_var = np.zeros(n_reg)   # output variance at each region

    # find the new region- and solution-layer index for all sampled solutions
    shift_sampled_vec = sampled_vec[:, shift_vector]       # 2d array, size = n2 * dim
    sol_vec_temp = shift_sampled_vec[:, :dim_sol]         # 2d array, size = n2 * dim_sol
    reg_vec_temp = shift_sampled_vec[:, dim_sol:]         # 2d array, size = n2 * dim_reg
    sampled_reg_index = np.ravel_multi_index(np.transpose(reg_vec_temp), grid_dim_reg)         # 1d array, size = n2
    sampled_sol_index = np.ravel_multi_index(np.transpose(sol_vec_temp), grid_dim_sol)         # 1d array, size = n2
    # create the tuple of sampled solutions for indexing
    sampled_index = (sampled_reg_index, sampled_sol_index)    # tuple of two 1d arrays

    ## record all solution output and variance
    all_sol_rep[sampled_index] = sampled_rep         # 2d array, size = n_reg * n_sol
    all_sol_output[sampled_index] = sampled_mean      # 2d array, size = n_reg * n_sol
    all_sol_var[sampled_index] = sampled_var      # 2d array, size = n_reg * n_sol

    for reg in set(sampled_reg_index):
        # check if there are at least two solutions sampled in each sampled region
        sampled_sol = np.argwhere(all_sol_rep[reg] != 0)    # n2 * 1 array ???????????????????????
        sampled_sol = sampled_sol.flatten()      # convert 2d array to 1d array
        # print('sampled_sol_in_region', reg, ':', sampled_sol)
        n_sampled_sol = len(sampled_sol)
        
        # if there are at least two solutions sampled, we conmupte the region-layer output and variance
        if n_sampled_sol >= n_init_sol:
            # print('n_sampled_sol', n_sampled_sol)
            sampled_output = all_sol_output[reg, sampled_sol]    # n_sampled_sol array
            # print('sampled_output', sampled_output)
            all_reg_output[reg] = statistics.mean(sampled_output)
            all_reg_var[reg] = statistics.variance(sampled_output) / n_sampled_sol * (1 - n_sampled_sol / n_sol) + statistics.mean(sampled_output) / n_sampled_sol

    # Find the sampled region index under the new projection (more than 2 solutions sampled)
    sampled_reg_index = np.argwhere(all_reg_output != 0)    # n2 * 1 array
    sampled_reg_index = sampled_reg_index.flatten()    # convert 2d array to 1d array

    # make sure there at least n_init_reg regions sampled
    # If there are less than n_init_reg regions sampled (more than 2 solutions)
    if len(sampled_reg_index) < n_init_reg:
        
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

                # check if this solution has been sampled before, if not, simulate it
                if all_sol_rep[design_reg_index[i], design_sol_index] == 0:
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

    
    ## MLE for region and solution layers GMRF hyperparameters

    # Find the sampled region index under the new projection (more than 2 solutions sampled)
    sampled_reg_index = np.argwhere(all_reg_output != 0)    # n2 * 1 array
    sampled_reg_index = sampled_reg_index.flatten()    # convert 2d array to 1d array

    # # choose at most 100 regions for MLE computing
    # if len(sampled_reg_index) > 100:
    #     selected_sampled_reg_index = random.sample(list(sampled_reg_index), 100)
    # else:
    #     selected_sampled_reg_index = sampled_reg_index
    
    # tau1 = [0.4/dim_reg for _ in range(dim_reg)]
    # reg_lik = lambda x: likelihood_new(x, tau1, grid_dim_reg, selected_sampled_reg_index, all_reg_output[selected_sampled_reg_index], np.diag(all_reg_var[selected_sampled_reg_index]), 0, 0)[0]
    # # bounds tau[0] > 1e-6
    # reg_bnds = [(1e-6, None)]    # tau[0]
    # # initial point tau = [1/var(all_reg_output[selected_sampled_reg_index])]
    # reg_xinit = [1/np.var(all_reg_output[selected_sampled_reg_index])]

    # # In minimize function, only 'SLSQP' and ’trust-constr’ methods can consider bounds and constraints
    # # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html for more details
    # time_start = time.time()
    # res = scipy.optimize.minimize(reg_lik, x0 = reg_xinit, method = 'SLSQP', bounds = reg_bnds, options = {'maxiter': 20})
    # time_end = time.time()

    # # record the region-layer MLE result
    # tau0 = res.x       # array
    # beta = likelihood_new(tau0, tau1, grid_dim_reg, selected_sampled_reg_index, all_reg_output[selected_sampled_reg_index], np.diag(all_reg_var[selected_sampled_reg_index]), 0, 0)[1]
    # tau = list(tau0) + tau1
    # print(tau)
    # print('Region-layer MLE time cost', time_end-time_start, 's')

    # # # region-layer hyperparameter
    # # # objective function is the likelihood_func
    # # reg_lik = lambda x: likelihood_fun(x, grid_dim_reg, selected_sampled_reg_index, all_reg_output[selected_sampled_reg_index], np.diag(all_reg_var[selected_sampled_reg_index]), 0, 0)[0]
    # # # constraint: tau[1] + ... + tau[dim_reg] <= 0.5
    # # reg_const = ({'type': 'ineq', 'fun': lambda tau: 0.5 - sum(tau[1:])})
    # # # bounds tau[0] > 1e-6, 1e-6 < tau[i] < 1
    # # reg_bnds = [(1e-6, None)]    # tau[0]
    # # reg_bnds.extend([(1e-6, 1) for _ in range(dim_reg)])
    # # # initial point tau = [1/var(all_reg_output[design_reg_index]), 0.4/dim_reg, ..., 0.4/dim_reg]
    # # reg_xinit = [1/np.var(all_reg_output[selected_sampled_reg_index])]
    # # reg_xinit.extend([0.4/dim_reg for _ in range(dim_reg)])
    
    # # # In minimize function, only 'SLSQP' and ’trust-constr’ methods can consider bounds and constraints
    # # # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html for more details
    # # time_start = time.time()
    # # res = scipy.optimize.minimize(reg_lik, x0 = reg_xinit, method = 'trust-constr', bounds = reg_bnds, constraints = reg_const, options = {'maxiter': 20, 'xtol': 1e-04})
    # # time_end = time.time()

    # # # record the region-layer MLE result
    # # tau = res.x       # array
    # # beta = likelihood_fun(tau, grid_dim_reg, selected_sampled_reg_index, all_reg_output[selected_sampled_reg_index], np.diag(all_reg_var[selected_sampled_reg_index]), 0, 0)[1]
    # # print('Region-layer MLE time cost', time_end-time_start, 's')


    # # solution-layer hyperparameter
    # # find the region that contains the smallest sample mean point
    # opt_val = np.min(all_sol_output[np.nonzero(all_sol_output)])
    # opt_sol_reg, opt_s = np.where(all_sol_output == opt_val)
    # temp_sol_output = all_sol_output[opt_sol_reg]    # get all sampled output in this region (including 0) -> 2d array
    # temp_sol_output = temp_sol_output.flatten()    # convert 2d to 1d

    # # check if there are at least 'n_init_sol' design points in this region, if not, simulate more
    # sampled_sol_in_opt_index = np.argwhere(temp_sol_output != 0)     # n2 * 1 array, find the sampled design solution index (not 0). 
    # # print('sampled_sol_in_opt_index', sampled_sol_in_opt_index)
    # sampled_sol_in_opt_index = sampled_sol_in_opt_index.flatten()      # convert 2d to 1d

    # if len(sampled_sol_in_opt_index) < n_init_sol:
    #     sol_limits = [[0, grid_dim_sol[i]] for i in range(dim_sol)]
    #     sol_limits = np.array(sol_limits)
    #     sol_sampling = LHS(xlimits = sol_limits, criterion = 'maximin')

    #     # ind2sub
    #     opt_sol_reg_vec = np.asarray(np.unravel_index(opt_sol_reg, grid_dim_reg))      # tuple -> dim_reg * 1 array
    #     opt_sol_reg_vec = opt_sol_reg_vec.flatten()        # 2d -> 1d
    #     # print('opt_sol_reg_vec', opt_sol_reg_vec)

    #     # initial design solutions by latin hypercube sampling 
    #     design_sol = sol_sampling(n_init_sol)   # sample n_initi_sol solutions
    #     design_sol = np.floor(design_sol)     # n_init_sol * dim_sol ndarray, each row represnets a solution, now 0->10
    #     design_sol = np.array(design_sol, dtype = np.int64)

    #     for j in range(n_init_sol):
    #         # convert the sub vector solution to index
    #         design_sol_index = np.ravel_multi_index(design_sol[j], grid_dim_sol)

    #         # check if this solution has been sampled before, if not, simulate it
    #         if all_sol_rep[opt_sol_reg, design_sol_index] == 0:
    #             # get the original solution for simulation
    #             design_sol_total = np.concatenate((design_sol[j], opt_sol_reg_vec)) # concatencate the region and solution layers
    #             orig_design_sol = np.zeros([1, dim], dtype = np.int32)   
    #             orig_design_sol[:,shift_vector] = design_sol_total    # use the shift vector to get the original design solution, now [0,10]

    #             # simulate the solution
    #             sim_output = obj_func(orig_design_sol, n_rep)   # [1, n_rep] ndarray, 
                
    #             # record the simulation output
    #             all_sol_rep[opt_sol_reg, design_sol_index] = n_rep
    #             all_sol_output[opt_sol_reg, design_sol_index] = statistics.mean(sim_output[0])
    #             all_sol_var[opt_sol_reg, design_sol_index] = statistics.variance(sim_output[0])/n_rep
    #             # print('reg', opt_sol_reg, 'sol', design_sol_index, 'output', all_sol_output[opt_sol_reg, design_sol_index], 'var', all_sol_var[design_reg_index[i], design_sol_index])

    #     # record the region-layer simulation output
    #     # find the sampled sol in this region 
    #     sampled_index = np.argwhere(all_sol_rep[opt_sol_reg] != 0)    # n_init_sol * 2 array, the first column are all 0
    #     sampled_index = sampled_index[:, 1].flatten()   # convert 2d array to 1d array
    #     # print('sampled_index', sampled_index)
    #     sampled_output = all_sol_output[opt_sol_reg, sampled_index]    # n_init_sol array
    #     all_reg_output[opt_sol_reg] = statistics.mean(sampled_output)
    #     all_reg_var[opt_sol_reg] = statistics.variance(sampled_output) / n_init_sol * (1 - n_init_sol / n_sol) + statistics.mean(sampled_output) / n_init_sol
    #     # print('reg', opt_sol_reg, 'output', all_reg_output[opt_sol_reg], 'var', all_reg_var[opt_sol_reg])
    #     sampled_sol_in_opt_index = sampled_index 

    # temp_sol_output = all_sol_output[opt_sol_reg, sampled_sol_in_opt_index]     # n2 array, sampled output at opt_sol_reg
    # temp_sol_var = all_sol_var[opt_sol_reg, sampled_sol_in_opt_index]    # n2 array, sampled output variance at opt_sol_reg

    # # # compute hyperparameter
    # # # since in solutions layer, the theta_s has linear relation, we have x -> dim_sol. last dim of x = 0.5 - tau0*x[0]/2k_s - sum(x[1:])
    # # # print('design_index', sampled_sol_in_opt_index)
    # # # print('output', temp_sol_output)
    # # sol_lik = lambda x: likelihood_fun(np.concatenate((x, np.array([0.5 - tau[0] * x[0] / (2*n_sol) - sum(x[1:])]))), grid_dim_sol, sampled_sol_in_opt_index, temp_sol_output, np.diag(temp_sol_var), beta, 1)[0]
    # # # constraints: tau[0]*theta[0]/2ks + theta[1] + ... + theta[dim_sol-1] <= 0.5
    # # temp_A = np.zeros([dim_sol, dim_sol])
    # # temp_A[0, 0] = tau[0] / (2*n_sol)
    # # for i in range(1, dim_sol):
    # #     temp_A[0, i] = 1
    # #     temp_A[i, i] = 1
    # # sol_const = scipy.optimize.LinearConstraint(temp_A, lb = 0, ub = 0.5, keep_feasible = True)
    # # #  sol_const = ({'type': 'ineq', 'fun': lambda theta: 0.5 - sum(theta[1:]) - tau[0] * theta[0] / (2*n_sol)})

    # # # bounds 1e-6 < theta[0] < ks/tau[0], 1e-6 < theta[i] < 1
    # # sol_bnds = [(1e-6, n_sol/tau[0])]    # theta[0]
    # # sol_bnds.extend([(1e-6, 1) for _ in range(dim_sol - 1)])
    # # # initial point theta = [ks/2tau[0], 0.4/dim_sol, ..., 0.4/dim_sol]
    # # sol_xinit = [n_sol / (2*tau[0])]
    # # sol_xinit.extend([0.2/dim_sol for _ in range(dim_sol - 1)])

    # # # solution layer MLE
    # # time_start = time.time()
    # # res = scipy.optimize.minimize(sol_lik, x0 = sol_xinit, method = 'trust-constr', bounds = sol_bnds, constraints = sol_const, options = {'maxiter': 20, 'xtol': 1e-05})
    # # time_end = time.time()

    # # # record the region-layer MLE result
    # # theta_sol = res.x       # array
    # # # add the last dimension of theta_sol from linear relation
    # # theta_sol = np.append(theta_sol, [0.5 - tau[0] * theta_sol[0] / (2*n_sol) - sum(theta_sol[1:])])
    # # # theta[0] = 1/\tilde{theta[0]}, we use \tilde{theta[0]} for linear relation, more details are in the paper 
    # # for i in range(dim_sol):
    # #     theta_sol[i + 1] = (0.5 - tau[0] * theta_sol[0] / (2*n_sol)) / dim_sol

    # theta_sol = [n_sol / (2*tau[0])]
    # theta_sol.extend([0.25/dim_sol for _ in range(dim_sol)])

    ## MLE for region and solution layers GMRF hyperparameters
    
    [beta, tau, theta_sol] = MLE(all_reg_output, all_reg_var, all_sol_output, all_sol_var, \
        grid_dim_reg, grid_dim_sol, sampled_reg_index, dim_reg, dim_sol, n_sol)
    
    # theta[0] = 1/\tilde{theta[0]}, we use \tilde{theta[0]} for linear relation, more details are in the paper 
    theta_sol[0] = 1/theta_sol[0]

    # initialize the region-layer precision matrix T
    reg_prec_mat = prec_mat_constr(tau, grid_dim_reg)

    ## record the GMRF hyperparameter
    theta = np.zeros(dim + 1)
    theta[0] = theta_sol[0]
    # get the theta at solution-layer dimensions
    temp_sol_shift = [shift_vector[i] + 1 for i in range(dim_sol)]
    theta[temp_sol_shift] = theta_sol[1:dim_sol+1]
    # compute the theta at region-layer dimensions
    for i in range(dim_reg):
        theta[shift_vector[dim_sol+i] + 1] = tau[0]*tau[i+1]/(n_sol*theta[0])
    

    return [shift_vector, theta, beta, tau, theta_sol, reg_prec_mat, \
        grid_dim_reg, grid_dim_sol, n_reg, n_sol, \
        all_sol_rep, all_sol_output, all_sol_var, all_reg_output, all_reg_var]





    

# # Example to test:
# side = 11   # number of feasible solutions in each dimension
# dim = 6     # number of dimensions
# # vector of the no. of feasible solutions in each dimension: [11 11 11 11 11 11]' 
# grid_dim = side * np.ones(dim, dtype = np.int32) 
# dim_reg = 3
# n_init_reg = 5
# n_init_sol = 5
# n_rep = 10
# obj_func = simulation_zakharov

# sampled_vec = np.array([[3,4,5,2,1,7], [3,2,6,5,4,7], [4,7,9,5,1,3], [0,5,5,6,7,2]])
# sampled_rep = np.array([10,10,10,10])
# sampled_mean = np.array([4.454, 3.5435, 76.564, 23.34])
# sampled_var = np.array([0.232, 0.555, 0.343, 0.567])

# projection_selection(n_init_reg, n_init_sol, n_rep, grid_dim, dim_reg, obj_func, sampled_vec, sampled_rep, sampled_mean, sampled_var)
    




