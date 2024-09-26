import numpy as np
from zakharov import *
    
def simulation_zakharov(x = None, r = None, generator = None): 

# This function adds a random error to the Branin function
# Example to test:
# x1 = np.array([[1,3,2,4,4,5], [2,3,3,2,4,1]])
# a = simulation_zakharov(x1, 10)
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
        x_vector = x_vector - 5
        for R in range(r):
            output[k, R] = zakharov(x_vector) + generator.normal(scale = 3.9, size = 1)
            # output[k, R] = zakharov(x_vector) + 3.9 * np.random.randn(1)
    
    return output
