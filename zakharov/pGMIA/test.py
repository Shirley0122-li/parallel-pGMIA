import numpy as np
from zakharov import *

sampled_temp = (np.array([  56,   56,   56,   56,  403,  403,  403,  403,  403,  403,  403,
        403,  403,  403,  403,  403,  403,  527,  527,  527,  527,  860,
        860,  860,  860, 1061, 1061, 1061, 1061, 1061, 1061, 1061, 1061,
       1061, 1061, 1061]), np.array([  38,  572,  811,  971,   77,  194,  260,  544,  683,  743,  913,
        965, 1003, 1016, 1087, 1246, 1261,  245,  646,  672, 1084,  107,
        368,  760, 1175,   18,  297,  306,  412,  659,  944, 1015, 1109,
       1238, 1239, 1282]))

print(type(sampled_temp))
new = np.transpose(sampled_temp)
print(type(new))
print(type(sampled_temp[0]))

side = 11   # number of feasible solutions in each dimension
dim = 6     # number of dimensions
# vector of the no. of feasible solutions in each dimension: [11 11 11 11 11 11]' 
grid_dim = side * np.ones(dim, dtype = np.int32) 
grid_dim_reg = grid_dim[3:6]
grid_dim_sol = grid_dim[0:3]
shift_vector = [1,2,5,0,3,4]
print(grid_dim_reg)

reg_index_temp = np.unravel_index(sampled_temp[0], grid_dim_reg)
sol_index_temp = np.unravel_index(sampled_temp[1], grid_dim_sol)
print(reg_index_temp)
print(sol_index_temp)
# design_reg_index[i] = np.ravel_multi_index(design_reg[i], grid_dim_reg)
# if want ind2sub
# temp = np.unravel_index(design_reg_index[i], grid_dim_reg)

total_index = np.transpose(np.concatenate((sol_index_temp, reg_index_temp)))
print(total_index)
print(type(total_index))

n2 = len(total_index)
sampled_index = np.zeros([n2, dim], dtype = np.int32)
sampled_index[:, shift_vector] = total_index
print(sampled_index)

print(zakharov([0,0,0,0,0,0]))