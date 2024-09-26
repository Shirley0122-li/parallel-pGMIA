import numpy as np
    
def branin(x = None): 
    
# Branin function
# The number of active dimensions n = 2.
# Example to test:
# branin(np.array([1,1])) -> y = 88.9041

# Input: 
# x (1 * d array): values in [0, 1], only the first 2 dimensions are active
# Process:
# 1. rescale: t = [(x1+5)/5, (x2-5)/5]
# 2. compute: y = ( x2-(5.1/4pi^2)x1^2 + 5x1/pi - 6 )^2 + 10(1-1/8pi)cos(x1) + 10
# Output:
# y = branin(x)
    
    # the LB, UB of two dimensions
    bounds = np.array([[-5,5],[0,10]])
    
    # rescale the first two dimensions for Branin computing
    rescaled_x = [0 for _ in range(2)]
    for i in range(2):
        rescaled_x[i] = bounds[i, 0] + x[i] * (bounds[i, 1] - bounds[i, 0])

    # use the formula to compute y
    y = (rescaled_x[1] - (5.1 / (4 * (np.pi ** 2))) * (rescaled_x[0] ** 2) + 5 * rescaled_x[0] / np.pi - 6) ** 2 + 10 * (1 - 1 / (8 * np.pi)) * np.cos(rescaled_x[0]) + 10
    return y

