import numpy as np

def zakharov(x):

# Zakharov function

# Input: 
# x (1d array, size = d): values in [-5, 5], only the first 2 dimensions are active
# Process:
# f(x) = \sum_{i=1}^d x_i^2 + (\sum_{i=1}^d 0.5 * i * x_i)^2 + (sum_{i=1}^d 0.5 * i * x_i)^4 
# Output:
# y = zakharov(x)

# Example to test:
# zakharov(np.array([1,3,2,4,-1,-1])) -> y = 6674

    d = len(x)    # dimension of input
    # sum1 = \sum_{i=1}^d x_i^2
    # sum2 = \sum_{i=1}^d 0.5 * i * x_i
    sum1, sum2 = 0, 0

    for i in range(d):
        xi = x[i]
        sum1 += xi ** 2
        sum2 += 0.5 * (i+1) * xi
    
    y = sum1 + sum2 ** 2 + sum2 ** 4

    return y
