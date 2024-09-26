from prec_mat_constr import *

def update_prec_mat(Q, theta, sol, reg_index, shift_vector, dim_reg, grid_dim):

# This function creates a new precision matrix for set S of solutions
# S = X(R) + tulde x   when tulde x is the neighbor of R

# Input:
# theta (dim+1 * 1 array): hyperparameters of single-layer GMRF
# sol (dim * 1 array): the vector format of a solution
# reg_index (scalar): the index of a region that we want to compare
# shift_vector (dim * 1 array): the shift vector under new projection
# dim_reg (scalar): number of dimensions in the region layer
# grid_dim (dim * 1 array): grid dimension 

# Output:
# new precision matrix (k_s+1 * k_s+1 array): new precision matrix of solutions in S.

# Example to test:
# theta = [1e-3, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05]
# sol = np.array([4,3,2,2,9,2])
# reg_index = 354 
# shift_vector = np.array([0,1,2,3,4,5]) 
# dim_reg = 3 
# grid_dim = np.array([11,11,11,11,11,11])
# print(check_if_neighbor(sol, reg_index, shift_vector, dim_reg, grid_dim_reg))

    # problem parameters
    dim = len(shift_vector)    # number of dimensions in the problem
    dim_sol = dim - dim_reg
    grid_dim_shift = grid_dim[shift_vector]  # shifted grid_dimension
    grid_dim_reg = grid_dim_shift[dim_sol:dim]  # grid_dim in the region layer
    grid_dim_sol = grid_dim_shift[0:dim_sol]   # grid_dim in the solution layer

    # create the original intrinsic solution-layer precision matrix
    theta_sol = np.array([1e-8, 0.1, 0.1, 0.1])
    prec_mat = Q

    # region of the solution
    shift_sol = sol[shift_vector]   # find the vector format of a solution under projection
    reg_of_sol = shift_sol[dim_sol:dim]    # find the region of this solution
    sol_of_sol = shift_sol[0:dim_sol]

    # region that want to compare
    reg_sub = np.unravel_index(reg_index, grid_dim_reg)
    # print('reg_sub', reg_sub, 'type')

    # check the dimension of solution to be the neighbor of region
    for i in range(dim_reg):
        if abs(reg_of_sol[i] - reg_sub[i]) == 1:
            break
    
    # compute the solution index within the region that is the neighbor of sol
    # convert the sub vector solution to index
    sol_index = np.ravel_multi_index(sol_of_sol, grid_dim_sol)
    
    # extend precision matrix
    # Get the shape of the existing array
    rows, columns = prec_mat.shape
    # Create a new array with an additional row and column of zeros
    updated_prec_mat = np.zeros((rows + 1, columns + 1))
    # Copy the existing array into the new array
    updated_prec_mat[:-1, :-1] = prec_mat

    # extract theta_i
    new_theta = 0.1
    val_in_mat = - theta_sol[0] * new_theta

    # fill in the parameter in updated precision matrix
    updated_prec_mat[-1, -1] = theta_sol[0]
    updated_prec_mat[-1, sol_index] = val_in_mat
    updated_prec_mat[-1, sol_index] = val_in_mat

    return updated_prec_mat

# theta = [1e-3, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05]
# sol = np.array([4,3,2,2,9,2])
# reg_index = 354 
# shift_vector = np.array([0,1,2,3,4,5]) 
# dim_reg = 3 
# grid_dim = np.array([11,11,11,11,11,11])
# print(new_prec_mat(theta, sol, reg_index, shift_vector, dim_reg, grid_dim))