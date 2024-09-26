from scipy.linalg import toeplitz
from scipy.linalg import triu
import numpy as np

def prec_mat_constr(theta, grid_dim):

# This function creates a sparse precision matrix
# Input:
# theta: a (d+1)-dim array, representing \theta_0, \theta_1, \dots, \theta_d
# grid_dim: a d-dim array, each element represents the number of feasible solutions at each dimensions
# Output: a precision matrix Q

# EXAMPLE to test
# theta = [1, 0.1, 0.2, 0.05]
# grid_dim = [3, 3, 3]
# a = prec_mat_constr(theta, grid_dim)
# print(a)
# l = np.linalg.cholesky(a)
# print(l)
    
    # number of dimensions for the problem
    d = len(grid_dim)   
    # number of feasible solutions
    n = np.prod(grid_dim)   
    
    # create a row (diag -> off-diag) for the toeplitz matrix for precision mat
    row = [0 for _ in range(n)]  
    # index for non-zero elements in row
    offset = [] 
    # first element in row
    row[0] = theta[0]

    # the rest of elements in row
    temp = 1
    offset.append(temp)  # record non-zero
    for i in range(1,d+1):
        row[temp] = -theta[0]*theta[i]
        temp *= grid_dim[i-1]
        offset.append(temp)
    
    # Toeplitz matrix 
    Q = toeplitz(row)

    # Take into account the boundary effects
    # We only change the upper triangle of Q.
    for i in range(d-1):
        # find the boudary points the d dimension
        # in the example, first time [3,6,9,12,15,18,21,24]
        #                 second time [9,18]
        end_zeroed_rows = np.arange(offset[i+1], n-1, offset[i+1])
        
        if i == 0:
            zeroed_rows = end_zeroed_rows
        else:
            # for higher dimensions, the boundary points are connected, such that we need begin->end
            # in the example, [7,16]
            begin_zeroed_rows = end_zeroed_rows - offset[i] + 1
            
            # create a vector of all begins->ends
            # in the example, [7,8,9,16,17,18]   (7->9, 16->18)
            zeroed_rows = []
            for j in range(len(end_zeroed_rows)):
                zeroed_rows.extend(range(begin_zeroed_rows[j],end_zeroed_rows[j]+1))
            # make it an np array    
            zeroed_rows = np.array(zeroed_rows)
        
        # set the boudary relations to be 0
        # -1 because Python starts from 0
        Q[zeroed_rows - 1,zeroed_rows + offset[i] - 1] = 0
    
    # do the same change on lower bound such that Q is the precision matrix
    Q = triu(Q) + np.transpose(triu(Q)) - theta[0]*np.identity(n)

    return Q
