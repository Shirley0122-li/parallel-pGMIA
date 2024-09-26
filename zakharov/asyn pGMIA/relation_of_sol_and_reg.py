import numpy as np

def check_if_neighbor(sol, reg_index, shift_vector, dim_reg, grid_dim_reg):

# This function computes if a solution is the neighbor of a region
# return 1 if True, 0 if False

# Input:
# sol (dim * 1 array): the vector format of a solution
# reg_index (scalar): the index of a region that we want to compare
# shift_vector (dim * 1 array): the shift vector under new projection
# dim_reg (scalar): number of dimensions in the region layer

# Output:
# 1/0 (boolean): 1 if is neighbor, 0 else.

# Example to test:
# sol = np.array([4,3,2,2,9,2])
# reg_index = 354 
# shift_vector = np.array([0,1,2,3,4,5]) 
# dim_reg = 3 
# grid_dim_reg = np.array([11,11,11])
# print(check_if_neighbor(sol, reg_index, shift_vector, dim_reg, grid_dim_reg))


    # region of the solution
    dim = len(shift_vector)    # number of dimensions in the problem
    dim_sol = dim - dim_reg    # number of dimensions in the solution layer
    shift_sol = sol[shift_vector]   # find the vector format of a solution under projection
    reg_of_sol = shift_sol[dim_sol:dim]    # find the region of this solution

    # region that want to compare
    reg_sub = np.unravel_index(reg_index, grid_dim_reg)
    # print('reg_sub', reg_sub, 'type')

    # check if solution is the neighbor of region
    # =>  if reg_of_sol is the neighbor of reg_sub
    test = 0
    for i in range(dim_reg):
        test += abs(reg_of_sol[i] - reg_sub[i])
    
    return test == 1



def check_if_in_reg(sol, reg_index, shift_vector, dim_reg, grid_dim_reg):

# This function computes if a solution is in the region
# return 1 if True, 0 if False

# Input:
# sol (dim * 1 array): the vector format of a solution
# reg_index (scalar): the index of a region that we want to compare
# shift_vector (dim * 1 array): the shift vector under new projection
# dim_reg (scalar): number of dimensions in the region layer

# Output:
# 1/0 (boolean): 1 if is neighbor, 0 else.

# Example to test:
# sol = np.array([4,3,2,2,9,2])
# reg_index = 354 
# shift_vector = np.array([0,1,2,3,4,5]) 
# dim_reg = 3 
# grid_dim_reg = np.array([11,11,11])
# print(check_if_in_reg(sol, reg_index, shift_vector, dim_reg, grid_dim_reg))


    # region of the solution
    dim = len(shift_vector)    # number of dimensions in the problem
    dim_sol = dim - dim_reg    # number of dimensions in the solution layer
    shift_sol = sol[shift_vector]   # find the vector format of a solution under projection
    reg_of_sol = shift_sol[dim_sol:dim]    # find the region of this solution

    # convert the sub vector solution to index
    reg_of_sol_index = np.ravel_multi_index(reg_of_sol, grid_dim_reg)
    
    return reg_of_sol_index == reg_index

