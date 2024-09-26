import numpy as np
import math
import random
import scipy.optimize   # package for finding GMRF hyperparameter
from likelihood_fun import *

def MLE(all_reg_output, all_reg_var, all_sol_output, all_sol_var, \
    grid_dim_reg, grid_dim_sol, sampled_reg_index, dim_reg, dim_sol, n_sol):

## This function computes the MLE for region and solution layers GMRF hyperparameters
# # Input:
# all_reg_output (1d array, size = n_reg)
# all_reg_var (1d array, size = n_reg)
# all_sol_output (2d array, size = n_reg * n_sol)
# all_sol_var (2d array, size = n_reg * n_sol)
# grid_dim_reg (1d array, size = dim_reg): grid_dim in the region layer
# grid_dim_sol (1d array, size = dim_sol): grid_dim in the solution layer 
# sampled_reg_index (1d array, size = n_init_reg): index of design regions 
# dim_reg (scalar): number of dimensions in the region layer
# dim_sol (scalar): number of dimensions in the solution layer
# n_sol (scalar): number of solutions in each region
# 
# Output:
# beta (scalar)
# tau (1d array, size = dim_reg + 1)
# theta_sol (1d array, size = dim_sol + 1)

# Example to test

# import pickle

# # open a file, where you stored the pickled data
# file = open('initial_output', 'rb')

# # dump information to that file
# data = pickle.load(file)
# [all_reg_output, all_reg_var, sampled_reg_index, all_sol_output, all_sol_var, \
#     grid_dim_reg, grid_dim_sol, sampled_reg_index, dim_reg, dim_sol, n_sol] = data

# # close the file
# file.close()

# MLE(all_reg_output, all_reg_var, all_sol_output, all_sol_var, \
#     grid_dim_reg, grid_dim_sol, sampled_reg_index, dim_reg, dim_sol, n_sol)

#########################################

    ## region-layer MLE

    # choose at most 100 regions for MLE computing
    if len(sampled_reg_index) > 100:
        sampled_reg_index = random.sample(list(sampled_reg_index), 100)
    else:
        sampled_reg_index = sampled_reg_index

    # sample a set of tau to test (tau_i > 0 and \sum_{i} tau_i = 1)
    n_tau_to_test = 10

    dist = [1.5 for _ in range(dim_reg)]
    tau_to_test = np.random.dirichlet(dist, n_tau_to_test)

    # sample a random variable in (0, 0.5) to determine the sum of tau
    # inversion sampling: F(x) = (2x)^{dim_reg}
    sums = np.random.uniform(0, 1, n_tau_to_test)
    sums = [pow(sums[i], 1/dim_reg)/2 for i in range(len(sums))]

    # tau_i = sum * tau_i (create tau whose sum is fixed < 0.5)
    tau_to_test = [tau_to_test[i] * sums[i] for i in range(n_tau_to_test)]

    # bounds tau[0] > 1e-6
    reg_bnds = [(1e-6, None)]    # tau[0]
    # initial point tau = [1/var(all_reg_output[sampled_reg_index])]
    reg_xinit = [1/np.var(all_reg_output[sampled_reg_index])]
    # print(reg_xinit)

    # compute the tau[0] corresponding to each tau, and choose the one with largest reg_lik
    cur_opt_l = math.inf
    try:
        for t_i in range(n_tau_to_test):
            temp_tau = tau_to_test[t_i]
            
            # In minimize function, only 'SLSQP' and ’trust-constr’ methods can consider bounds and constraints
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html for more details
            reg_lik = lambda x: likelihood_new(x, temp_tau, grid_dim_reg, sampled_reg_index, all_reg_output[sampled_reg_index], np.diag(all_reg_var[sampled_reg_index]), 0, 0)[0]
            res = scipy.optimize.minimize(reg_lik, x0 = reg_xinit, method = 'SLSQP', bounds = reg_bnds, options = {'maxiter': 20})
            
            likelihood = likelihood_new(res.x, temp_tau, grid_dim_reg, sampled_reg_index, all_reg_output[sampled_reg_index], np.diag(all_reg_var[sampled_reg_index]), 0, 0)[0]
            temp_beta = likelihood_new(res.x, temp_tau, grid_dim_reg, sampled_reg_index, all_reg_output[sampled_reg_index], np.diag(all_reg_var[sampled_reg_index]), 0, 0)[1]
            if likelihood < cur_opt_l:
                cur_opt_l = likelihood
                tau0 = res.x
                tau = list(tau0) + list(temp_tau)
                beta = temp_beta
                # print(tau0)
    except:
        # print('oh no')
        tau0 = [1e-4]
        temp_tau = [tau_to_test[0]]
        tau = list(tau0) + list(temp_tau)
        beta = np.mean(all_reg_output[sampled_reg_index])

    # print('Region-layer MLE finished')

    ## solution-layer MLE (hyperparameter)

    # find the region that contains the smallest sample mean point
    opt_val = np.min(all_sol_output[np.nonzero(all_sol_output)])
    opt_sol_reg, opt_s = np.where(all_sol_output == opt_val)
    temp_sol_output = all_sol_output[opt_sol_reg]    # get all sampled output in this region (including 0) -> 2d array
    temp_sol_output = temp_sol_output.flatten()    # convert 2d to 1d

    design_sol_index = np.argwhere(temp_sol_output != 0)     # n2 * 1 array, find the sampled design solution index (not 0). 
    design_sol_index = design_sol_index.flatten()      # convert 2d to 1d
    temp_sol_output = all_sol_output[opt_sol_reg, design_sol_index]     # n2 array, sampled output at opt_sol_reg
    temp_sol_var = all_sol_var[opt_sol_reg, design_sol_index]    # n2 array, sampled output variance at opt_sol_reg

    # compute solution-layer hyperparameter  (dim_sol)
    # sample a few numbers of solution-layer hyperparmeter to test
    # because of the linear constraint, we only sample 'dim_sol - 1' dimensions
    # sample a set of theta to test (theta_i > 0 and \sum_{i} theta_i = 1)
    n_theta_to_test = 10

    dist = [1.5 for _ in range(dim_sol - 1)]
    theta_to_test = np.random.dirichlet(dist, n_theta_to_test)

    # sample a random variable in (0, 0.5) to determine the sum of theta
    # inversion sampling: F(x) = (2x)^{dim_reg}
    sums = np.random.uniform(0, 1, n_theta_to_test)
    sums = [pow(sums[i], 1/(dim_sol-1))/2 for i in range(len(sums))]

    # theta_i = sum * theta_i (create theta whose sum is fixed < 0.5)
    theta_to_test = [theta_to_test[i] * sums[i] for i in range(n_theta_to_test)]

        # find the opt solution-layer by likeihood
    cur_opt_l = math.inf
    for t_i in range(len(theta_to_test)):
        
        temp_theta = theta_to_test[t_i]
        # print('temp_theta', temp_theta, 'tau0', tau0)

        # since in solutions layer, the theta_s has linear relation, we have x -> dim_sol. last dim of x = 0.5 - tau0*x[0]/2k_s - sum(x[1:])
        # constraints: 0 <= tau[0]*theta[0]/2ks <= 0.5 - (theta[1] + ... + theta[dim_sol-1])
        temp_A = tau[0] / (2*n_sol)
        sol_bnds = [(1e-6, (0.499 - sum(temp_theta))/temp_A)]
        sol_xinit = [(0.499 - sum(temp_theta))/(2*temp_A)]
        # print('sol_xinit', sol_xinit)
        # sol_const = scipy.optimize.LinearConstraint(temp_A, lb = 0, ub = 0.5 - sum(temp_theta), keep_feasible = True)
        # In minimize function, only 'SLSQP' and ’trust-constr’ methods can consider bounds and constraints
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html for more details
        x = sol_xinit[0]
        sol_const = scipy.optimize.LinearConstraint(1, lb = 1e-6, ub = (0.499 - sum(temp_theta))/temp_A, keep_feasible = True)

        try:
            sol_lik = lambda x: likelihood_new_sol(x, temp_theta, grid_dim_sol, design_sol_index, temp_sol_output, np.diag(temp_sol_var), beta, temp_A)
            # res = scipy.optimize.minimize(sol_lik, x0 = sol_xinit, method = 'SLSQP', bounds = sol_bnds, options = {'maxiter': 20})
            res = scipy.optimize.minimize(sol_lik, x0 = sol_xinit, method = 'SLSQP', bounds = sol_bnds, constraints = [sol_const], options = {'maxiter': 20})
            
            theta_last = 0.5 - temp_A * res.x - sum(temp_theta)
            likelihood = likelihood_new_sol(res.x, temp_theta, grid_dim_sol, design_sol_index, temp_sol_output, np.diag(temp_sol_var), beta, temp_A)
            
            theta_sol0 = res.x
            temp = list(theta_sol0) + list(temp_theta) + list(theta_last)
            # print('solution-layer hyperparameter:', temp)
        except:
            likelihood = math.inf
            theta_sol0 = [1e-4]
            theta_last = [0.499 - sum(temp_theta)]
            # print(theta_sol0, 'temp', temp_theta, 'theta_last', theta_last)
            theta_sol = list(theta_sol0) + list(temp_theta) + list(theta_last)

        if likelihood < cur_opt_l:
            cur_opt_l = likelihood
            theta_sol0 = res.x
            theta_sol = list(theta_sol0) + list(temp_theta) + list(theta_last)
            # print('solution-layer hyperparameter:', theta_sol)

    return [beta, tau, theta_sol]

# import pickle

# # open a file, where you stored the pickled data
# file = open('initial_output', 'rb')

# # dump information to that file
# data = pickle.load(file)
# [all_reg_output, all_reg_var, sampled_reg_index, all_sol_output, all_sol_var, \
#     grid_dim_reg, grid_dim_sol, sampled_reg_index, dim_reg, dim_sol, n_sol] = data

# # close the file
# file.close()

# res = MLE(all_reg_output, all_reg_var, all_sol_output, all_sol_var, \
#     grid_dim_reg, grid_dim_sol, sampled_reg_index, dim_reg, dim_sol, n_sol)
# # print(res)
