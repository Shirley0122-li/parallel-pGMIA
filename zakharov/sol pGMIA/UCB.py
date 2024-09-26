import numpy as np
import math

def UCB(opt_index, cond_mean, cond_var, n_sampled_nodes):

# This function computes the complete expected improvement at all 
# solutions given the current optimal
# UCB(x, a) = M(x) - sqrt(alpha) * V(x)

# Input:
# opt_index (scalar): current optimal index
# cond_mean (n * 1 array): conditional means at all solutions
# cond_var (n * 1 array): conditional variances at all solutions
# n_sampled_nodes (scalar): number of sampled region-layer nodes, used to compute alpha

# Output:
# ucb (n * 1 array): upper confidence bound at all solutions

# Example to test:
# opt_index = 2
# cond_mean = np.array([1,1,0.5,1,1,1,1])
# cond_var = np.array([0.1, 0.1, 0.2, 0.2, 0.3, 0.4, 0.5])
# cond_cov = np.array([0.02, 0.02, 0.02, 0.04, 0.05, 0.04, 0.09])
# ucb_value = UCB(opt_index, cond_mean, cond_var, cond_cov)
# print(ucb_value)

    # pick delta value
    # delta in (0,1) is a user-defined parameter \citep{srinivas2009}. 
    delta = 1

    # compute the value of alpha
    # The larger delta is, the larger alpha becomes, which induces more exploration.
    alpha = 2 * np.log(np.square(n_sampled_nodes) / np.sqrt(len(cond_mean)) * np.square(math.pi) / (6 * delta))
    if alpha < 0:
        alpha = 1e-10

    # prevents negative variances obtained through comp errors
    # print(type(cond_var))
    # print(cond_var < 1e-10)

    # compute ucb from the formula
    ucb = cond_mean - np.sqrt(alpha) * np.sqrt(cond_var)

    # return -ucb such that we are still maximizing the sampling criterion
    return -ucb

# opt_index = 2
# cond_mean = np.array([1,1,0.5,1,1,1,1])
# cond_var = np.array([0.1, 0.1, 0.2, 0.2, 0.3, 0.4, 0.5])
# cond_cov = np.array([0.02, 0.02, 0.02, 0.04, 0.05, 0.04, 0.09])
# n_sampled_nodes = 3
# ucb_value = UCB(opt_index, cond_mean, cond_var, cond_cov, n_sampled_nodes)
# print(ucb_value)