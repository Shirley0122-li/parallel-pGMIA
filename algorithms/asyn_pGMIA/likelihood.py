import numpy as np
import math
from prec_mat_constr import *

def likelihood_fun(theta, grid_dim, sampled_sol_index, sampled_output, Q_e_inv, beta, layer):

# This function computes the log likelihood given the data and parameters
# Input:
# theta (d+1 array): parameters for GMRF precision matrix
# grid_dim (d array): the problem of interest
# sampled_sol_index (n_2 array): the index of sampled solutions
# sampled_output (n_2 array): the simulation output at all sampled solutions
# Q_e_inv (n_2 * n_2 array): the simulation output variance at all sampled solutions
# beta (scalar): if in the solution layer, the beta is given
# layer: 0 represents single-layer or region-layer, 1 represents solution layer 
# 
# Output:
# [0] -> log likelihood given the data and parameters
# [1] -> beta (scalar): mean parameter of GMRF

# Example to test
# theta = [1, 0.1, 0.2, 0.05]
# grid_dim = [3, 3, 3]
# sampled_sol_index = np.array([2, 5])
# sampled_output = np.array([2.0, 3.0])
# Q_e_inv = [[0.01, 0], [0, 0.02]]

# likelihood_fun(theta, grid_dim, sampled_sol_index, sampled_output, Q_e_inv, 0, 0)

    # if it's in solution layer, theta[0] = 1/theta[0]

    if layer == 1:
        theta[0] = 1/theta[0]

    # find n & n_2
    n = int(np.prod(grid_dim))
    n2 = len(sampled_sol_index)

    # create identity matrix of size n and precision matrix
    I = np.identity(n)
    Q = prec_mat_constr(theta, grid_dim)

    # cholesky factorization (lower)
    L_Q = np.linalg.cholesky(Q)  # cholesky works well
    Sigma_22 = np.linalg.solve(np.transpose(L_Q), np.linalg.solve(L_Q, I[:, sampled_sol_index]))
    Sigma_22 = Sigma_22[sampled_sol_index, :]
    T = (Sigma_22 + Q_e_inv) * theta[0]

    # compute mean parameter in region layer
    if layer == 0:
        temp1 = np.linalg.solve(Sigma_22 + Q_e_inv, np.ones((n2, 1)))   # n2 * 1
        temp2 = np.linalg.solve(Sigma_22 + Q_e_inv, sampled_output)   # 1 * n2
        beta = np.linalg.solve(np.ones((1, n2)) @ temp1, np.ones((1, n2)) @ temp2)   # 1 * 1
        beta = beta[0]

    # negative loglikelihood
    temp3 = np.linalg.solve(Sigma_22 + Q_e_inv, sampled_output - np.ones(n2))
    L_neg = - 0.5 * n2 * math.log(theta[0]) + 0.5 * np.prod(np.linalg.slogdet(T)) + 0.5 * (sampled_output - beta * np.ones(n2)) @ temp3

    return [L_neg, beta]

def likelihood_new(theta0, theta, grid_dim, sampled_sol_index, sampled_output, Q_e_inv, beta, layer):

# This function computes the log likelihood given the data and parameters
# Input:
# theta0 (scalar): parameter for GMRF
# theta (d array): parameters for GMRF precision matrix
# grid_dim (d array): the problem of interest
# sampled_sol_index (n_2 array): the index of sampled solutions
# sampled_output (n_2 array): the simulation output at all sampled solutions
# Q_e_inv (n_2 * n_2 array): the simulation output variance at all sampled solutions
# beta (scalar): if in the solution layer, the beta is given
# layer: 0 represents single-layer or region-layer, 1 represents solution layer 
# 
# Output:
# [0] -> log likelihood given the data and parameters
# [1] -> beta (scalar): mean parameter of GMRF

# Example to test
# theta = [1, 0.1, 0.2, 0.05]
# grid_dim = [3, 3, 3]
# sampled_sol_index = np.array([2, 5])
# sampled_output = np.array([2.0, 3.0])
# Q_e_inv = [[0.01, 0], [0, 0.02]]

# likelihood_fun(theta, grid_dim, sampled_sol_index, sampled_output, Q_e_inv, 0, 0)

    # if it's in solution layer, theta0 = 1/theta0
    if layer == 1:
        theta0 = 1/theta0
    
    theta0 = np.take(theta0,0)

    # find n & n_2
    n = int(np.prod(grid_dim))
    n2 = len(sampled_sol_index)

    # create identity matrix of size n and precision matrix
    I = np.identity(n)
    # print(theta0, 'theta', theta)
    theta = [theta0] + list(theta)
    Q = prec_mat_constr(theta, grid_dim)

    # cholesky factorization (lower)
    L_Q = np.linalg.cholesky(Q)  # cholesky works well
    Sigma_22 = np.linalg.solve(np.transpose(L_Q), np.linalg.solve(L_Q, I[:, sampled_sol_index]))
    Sigma_22 = Sigma_22[sampled_sol_index, :]
    T = (Sigma_22 + Q_e_inv) * theta0

    # compute mean parameter in region layer
    if layer == 0:
        temp1 = np.linalg.solve(Sigma_22 + Q_e_inv, np.ones((n2, 1)))   # n2 * 1
        temp2 = np.linalg.solve(Sigma_22 + Q_e_inv, sampled_output)   # 1 * n2
        beta = np.linalg.solve(np.ones((1, n2)) @ temp1, np.ones((1, n2)) @ temp2)   # 1 * 1
        beta = beta[0]

    # negative loglikelihood
    temp3 = np.linalg.solve(Sigma_22 + Q_e_inv, sampled_output - np.ones(n2))
    L_neg = - 0.5 * n2 * math.log(theta0) + 0.5 * np.prod(np.linalg.slogdet(T)) + 0.5 * (sampled_output - beta * np.ones(n2)) @ temp3

    return [L_neg, beta]

# theta0 = 1
# theta = [0.1, 0.2, 0.05]
# grid_dim = [3, 3, 3]
# sampled_sol_index = np.array([2, 5])
# sampled_output = np.array([2.0, 3.0])
# Q_e_inv = [[0.01, 0], [0, 0.02]]

# likelihood_new(theta0, theta, grid_dim, sampled_sol_index, sampled_output, Q_e_inv, 0, 0)

def likelihood_new_sol(theta0, theta, grid_dim, sampled_sol_index, sampled_output, Q_e_inv, beta, temp_A):

# This function computes the log likelihood given the data and parameters
# Input:
# theta0 (scalar): parameter for GMRF
# theta (d-1 array): parameters for GMRF precision matrix. Because it's solution layer, the last dimension can be computed
# grid_dim (d array): the problem of interest
# sampled_sol_index (n_2 array): the index of sampled solutions
# sampled_output (n_2 array): the simulation output at all sampled solutions
# Q_e_inv (n_2 * n_2 array): the simulation output variance at all sampled solutions
# beta (scalar): if in the solution layer, the beta is given
# 
# Output:
# [0] -> log likelihood given the data and parameters
# [1] -> beta (scalar): mean parameter of GMRF

# Example to test
# theta = [1, 0.1, 0.2, 0.05]
# grid_dim = [3, 3, 3]
# sampled_sol_index = np.array([2, 5])
# sampled_output = np.array([2.0, 3.0])
# Q_e_inv = [[0.01, 0], [0, 0.02]]

# likelihood_fun(theta, grid_dim, sampled_sol_index, sampled_output, Q_e_inv, 0, 0)

    # compute the last dimension corresponding to the parameter
    theta_last = 0.5 - temp_A * theta0 - sum(theta)
    # print('temp_A', temp_A)
    # print('bendan! sum should be 1:', 2 * sum(theta) + 2 * theta_last + temp_A *2 * theta0)
    # print('theta:', theta, ' last', theta_last, ' theta_shape:', theta.shape, ' last_shape', theta_last.shape)
    # print('bound1:', (0.5 - sum(theta))/temp_A, 'bound2:', 1/(2*temp_A))
    # print('theta0', theta0)
    theta = np.concatenate((theta, np.array(theta_last)))
    # print('theta:', theta)
    # print('sum should be 1:', 2 * sum(theta) + temp_A *2 * theta0)

    # in solution layer, theta0 = 1/theta0
    theta0 = 1/theta0
    theta0 = np.take(theta0,0)

    # find n & n_2
    n = int(np.prod(grid_dim))
    n2 = len(sampled_sol_index)

    # create identity matrix of size n and precision matrix
    I = np.identity(n)
    theta = [theta0] + list(theta)
    Q = prec_mat_constr(theta, grid_dim)

    # cholesky factorization (lower)
    L_Q = np.linalg.cholesky(Q)  # cholesky works well
    Sigma_22 = np.linalg.solve(np.transpose(L_Q), np.linalg.solve(L_Q, I[:, sampled_sol_index]))
    Sigma_22 = Sigma_22[sampled_sol_index, :]
    T = (Sigma_22 + Q_e_inv) * theta0

    # negative loglikelihood
    temp3 = np.linalg.solve(Sigma_22 + Q_e_inv, sampled_output - np.ones(n2))
    L_neg = - 0.5 * n2 * math.log(theta0) + 0.5 * np.prod(np.linalg.slogdet(T)) + 0.5 * (sampled_output - beta * np.ones(n2)) @ temp3

    return L_neg

# theta0 = 1
# theta = [0.1, 0.2, 0.05]
# grid_dim = [3, 3, 3]
# sampled_sol_index = np.array([2, 5])
# sampled_output = np.array([2.0, 3.0])
# Q_e_inv = [[0.01, 0], [0, 0.02]]

# likelihood_new(theta0, theta, grid_dim, sampled_sol_index, sampled_output, Q_e_inv, 0, 0)