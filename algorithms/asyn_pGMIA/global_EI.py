import numpy as np
from scipy.stats import norm
import random

def global_EI(opt_index, cond_mean, cond_var, cond_cov, x_opt_cond_mean, x_opt_cond_var):

# This function computes 
# 
# the complete expected improvement at all 
# solutions given the current optimal
# CEI(x, x~) = (M(x~) - M(x)) * Phi((M(x~) - M(x))/sqrt(V(x, x~))) 
#               + sqrt(V(x, x~)) * phi((M(x~) - M(x))/sqrt(V(x, x~)))

# Input:
# opt_index (scalar): current optimal index
# cond_mean (n * 1 array): conditional means at all solutions
# cond_var (n * 1 array): conditional variances at all solutions
# cond cov (n * 1 array): conditional covariances at all solutions
# x_opt_cond_mean (scalar): conditional mean at global optimum
# x_opt_cond_var (scalar): contiaional variance at global optimum

# Output:
# cei (n * 1 array): complete expected improvement at all solutions

# Example to test:
# opt_index = 2
# cond_mean = np.array([1,1,0.5,1,1,1,1])
# cond_var = np.array([0.1, 0.1, 0.2, 0.2, 0.3, 0.4, 0.5])
# cond_cov = np.array([0.02, 0.02, 0.02, 0.04, 0.05, 0.04, 0.09])
# cei_value = CEI(opt_index, cond_mean, cond_var, cond_cov)
# print(cei_value)

    # Find the conditonal mean and variance at the current optimal solution
    # x_opt_cond_mean = cond_mean[opt_index]
    # x_opt_cond_var = cond_var[opt_index]

    # V(x, x~) = V(x)
    var_of_dif = cond_var # under this case C(x, x~) = 0

    # M(x~) - M(x)
    dif_of_mean = x_opt_cond_mean - cond_mean

    # (M(x~) - M(x))/sqrt(V(x, x~))
    # since var_of_dif[opt_index] = 0 cannot be divided, set it to be 1e06 to avoid the invalid value error
    crit_val = dif_of_mean / np.sqrt(abs(var_of_dif))

    # EI
    ei = dif_of_mean * norm.cdf(crit_val) + np.sqrt(abs(var_of_dif)) * norm.pdf(crit_val)
    if sum(ei) == 0:
        temp = random.randint(0, len(ei) - 1)
        ei[temp] = 1
    
    return ei

