#  opt    second opt
# 0.6445   0.8685
# (0.8685 - 0.6445)/4*sqrt(30) ~ 0.3
import numpy as np
from branin import *
    
def simulation_branin(x = None,r = None): 

# This function adds a random error to the Branin function
# Example to test:
# x1 = np.array([[5,4], [3,8]])
# a = simulation_branin(x1, 10)
# print(a)

# Input:
# x (m * d array): feasible solutions to test, each row is a solution
# r (scalar): number of replications when we first visit a 
# Output:
# y (m * r array): sample outputs at m solutions with r replications

    # Create an empty output with fixed size 
    output = np.zeros([len(x), r])

    for k in range(len(x)):
        x_vector = x[k, :]
        x_vector = x_vector / 10
        for R in range(r):
            output[k, R] = branin(x_vector) + 0.6 ** 2 * np.random.randn(1)
    
    return output

