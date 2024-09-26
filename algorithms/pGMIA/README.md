# pGMIA-parallel
The main function is: GMIA_learning(NAI, max_iter)

Input "NAI" is the random seed, and input "max_iter" is the budget for pGMIA.

Example run: GMIA_learning(1, 100)

fcns/FineLevel_GMRF does the exploration on the soultion layer, which is the function that we want to parallel.

## GMIA_learning.m 

It includes the initilization of algorithm (Line 1-241), 
region-layer exploration at each iteration (Line 244-413), 
and projection updating (Line 415 - 585).

The result is saved in ".mat" every iteration.

## fcns/complete_expected_improvement.m

Let $n$ be the number of feasible solutions at each layer (region layer: $n = k_r$, and solution layer: $n = k_s$).
It computes the CEI in the region and solution layers given posterior mean, vairance and covariance.

Input: current optimal (1*1 scalar), posterior mean ($n$*1 vector), variance ($n$*1 vector), and covariance ($n$*1 vector).

Output: CEI value at all feasible solutions ($n$*1 vector).

## fcns/FineLevel_GMRF.m

Within each selected region, it finds two solutions to simulate (current optimal $\tilde{x}$ and solution with max CEI value $x_{CEI}$).

Input: all_point_output ($k_rk_s$*1 vector), all point output variance ($k_rk_s$*1 vector), all region output ($k_r$*1 vector), all region output variance ($k_r$*1 vector).   

Output: all_point_output ($k_rk_s$*1 vector), all point output variance ($k_rk_s$*1 vector), all region output ($k_r$*1 vector), all region output variance ($k_r$*1 vector).

----> This will be updated such that only $k_s$*1 vectors work as input and output.

## fcns/fminsearchcon.m

This is a function for minimization problem with constraints. Used for finding MLE.

## fcns/likelihood_fun.m and likelihood_fun_solution.m

The likelihood functions of GMRF parameters on the region and solution layers.

## fcns/logdet.m

Computation of logarithm of determinant of a matrix. Used in likelihood functions.

## fcns/prec_mat_constr.m

Construct the sparse precision matrix for GMRF given its parameters and problem size.

## fcns/projection_choosing.m

Given the current sampled result (output and variance), it selects a projection based on projecton selection criterion and return the region-layer GMRF parameters based on that.

Input: sampled_vector ($n_2$*d vector), sampled_output ($n_2$*1 vector), sampled_variance ($n_2$*1 vector). Here $n_2$ is the number of sampled solutions.

Output: GRMF parapeters: new_tau, new_beta. New projection: shift_vector (d*1 vector)

## fcns/sptoeplitz.m

It is used in prec_mat_constr.m to help constructing the precision matrix.

## testing

The testing folder includes example functions we applied in the algrithm. Including Branin, griewank, stybtang, ATO, etc.
