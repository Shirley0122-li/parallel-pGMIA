## import packages
import random
import os
from time import sleep
import numpy as np
import statistics
import time
import pickle
# package for LHS initial sampling
from smt.sampling_methods import LHS
# package for parallelizing
from mpi4py import MPI
# logging
import sys

## import self-defined functions
# simulation function
from simulation_zakharov import *
from likelihood import likelihood_fun 
from likelihood import likelihood_new
from prec_mat_constr import *
from CEI import *
from FineLevel_GMRF import *
from projection import projection_selection
from projection import projection_data_update
from MLE import MLE

##########################################################################################
# define MPI message tags
def enum(*sequential, **named):
	enums = dict(zip(sequential, range(len(sequential))), **named)
	return type('Enum', (), enums)

# Master sends 4 types of messages to workers.
# 1. SIMULATE_NEW (solution-layer exploration, including necessary data)
# 2. CONTINUE_OLD (continue doing job)
# 3. QUIT_JOB (quit jobs on the server)
# 4. TERMINATE_ALG (terminate all the workers)

# Worker sends 2 types of messages to master.
# 1. FINISH (this region stops its work because the stopping criterion is satisfied)
# 2. UPDATE (this region provides its data to the master for update)

tags = enum("SIMULATE_NEW", "CONTINUE_OLD", "QUIT_JOB", "TERMINATE_ALG", "FINISH", "UPDATE")

##########################################################################################

class ProblemParameter:
    def __init__(self):
        
        # objective function
        self.obj_func = simulation_zakharov   # Objective function used for simulation
        self.true_obj_func = zakharov   # True output of the objective function

        # size of problem
        self.dim = 6   # Total number of dimensions in the problem
        self.side = 11   # Length of each side of the hypercube (input space)
        self.dim_reg = 3   # Number of dimensions in the region layer

        # problem initial setting
        self.n_rep = 10    # number of replications when visiting a solution
        self.n_init_reg = 10  # number of initial design regions
        self.n_init_sol = 10  # number of initial design solutions within each design region
        self.projection_update_period = 10    # period for pGMIA to update the projection

##########################################################################################

class DataContainer:
    def __init__(self, side, dim, dim_reg, shift_vector):

        # vector of the no. of feasible solutions in each dimension: [11 11 11 11 11 11]' 
        self.grid_dim = side * np.ones(dim, dtype = np.int32) 

        ## projection initilization
        self.dim = dim   # Total number of dimensions in the problem
        self.dim_reg = dim_reg   # Number of dimensions in the region layer
        self.dim_sol = dim - dim_reg   # Number of solution-layer dimensions
        self.shift_vector = shift_vector   # Vector that shifts the position of the region-layer dimensions in the design
        
        ## dataset initialization
        # initialization of projection
        self.grid_dim_shift = self.grid_dim[shift_vector]  # shifted grid_dimension
        self.grid_dim_reg = self.grid_dim_shift[dim_reg:dim]  # grid_dim in the region layer
        self.grid_dim_sol = self.grid_dim_shift[0:dim_reg]   # grid_dim in the solution layer
        self.n_reg = int(np.prod(self.grid_dim_reg))   # number of feasible regions in the region layer
        self.n_sol = int(np.prod(self.grid_dim_sol))   # number of feasible solutions in each region
        
        # single-layer -> all solutions
        # [i,j] of the following matrices represents the jth solution at ith layer
        # NOTICE: np.empty does not set the array values to zero, and may therefore by marginally faster, should be used with caution!!
        self.all_sol_rep = np.zeros([self.n_reg, self.n_sol])    # number of replications at each solution
        self.all_sol_output = np.zeros([self.n_reg, self.n_sol])   # output mean at each solution
        self.all_sol_var = np.zeros([self.n_reg, self.n_sol])    # output variance at each solution
        self.count = np.zeros([self.n_reg, 1])        # number of a region visited

        # region-layer -> all region-layer nodes
        # the [i] of the following vector represents the ith region
        self.all_reg_output = np.zeros(self.n_reg)  # output mean at each region
        self.all_reg_var = np.zeros(self.n_reg)     # output variance at each region
    
    ##########################################################################################
    def pGMIA_initial_sample(self, prob, master_generator):
        """
        Simulate the initial design regions and design solutions.
        """
        ## simulate the initial design regions and design solutions
        # initial design regions by latin hypercube sampling
        # Set up region and solution limits
        reg_bounds = np.array([[0, self.grid_dim_reg[i]] for i in range(self.dim_reg)])
        sol_bounds = np.array([[0, self.grid_dim_sol[i]] for i in range(self.dim_sol)])
        
        # Initialize Latin Hypercube Sampling objects
        reg_sampling = LHS(xlimits=reg_bounds)
        sol_sampling = LHS(xlimits=sol_bounds)
        
        # Sample n_init_reg regions
        design_reg = reg_sampling(prob.n_init_reg)   # sample n_init_reg regions
        design_reg = np.floor(design_reg)   # n_init_reg * dim_reg ndarry, each row represents a region, now 0->10
        design_reg = np.array(design_reg, dtype = np.int64)
        
        # convert the sub vector of region to index 
        design_reg_index = np.empty(prob.n_init_reg, dtype = np.int64)
        for i in range(prob.n_init_reg):
            # sub2ind
            design_reg_index[i] = np.ravel_multi_index(design_reg[i], self.grid_dim_reg)
            # if want ind2sub
            # temp = np.unravel_index(design_reg_index[i], grid_dim_reg)
            
            # initial design solutions by latin hypercube sampling 
            design_sol = sol_sampling(prob.n_init_sol)   # sample n_initi_sol solutions
            design_sol = np.floor(design_sol)     # n_init_sol * dim_sol ndarray, each row represnets a solution, now 0->10
            design_sol = np.array(design_sol, dtype = np.int64)


            for j in range(prob.n_init_sol):
                # convert the sub vector solution to index
                design_sol_index = np.ravel_multi_index(design_sol[j], self.grid_dim_sol)

                # get the original solution for simulation
                design_sol_total = np.concatenate((design_sol[j], design_reg[i])) # concatencate the region and solution layers
                orig_design_sol = np.zeros([1, prob.dim], dtype = np.int32)   
                orig_design_sol[:,self.shift_vector] = design_sol_total    # use the shift vector to get the original design solution, now [0,10]

                # simulate the solution
                sim_output = prob.obj_func(orig_design_sol, prob.n_rep, master_generator)   # [1, n_rep] ndarray, 
                
                # record the simulation output
                self.all_sol_rep[design_reg_index[i], design_sol_index] = prob.n_rep
                self.all_sol_output[design_reg_index[i], design_sol_index] = statistics.mean(sim_output[0])
                self.all_sol_var[design_reg_index[i], design_sol_index] = statistics.variance(sim_output[0])/prob.n_rep

            # record the region-layer simulation output
            # find the sampled sol in this region 
            sampled_index = np.argwhere(self.all_sol_rep[design_reg_index[i]] != 0)    # n_init_sol * 1 array
            sampled_index = sampled_index.flatten()   # convert 2d array to 1d array
            sampled_output = self.all_sol_output[design_reg_index[i], sampled_index]    # n_init_sol array
            sampled_output_var = self.all_sol_var[design_reg_index[i], sampled_index]   # n_init_sol array
            self.all_reg_output[design_reg_index[i]] = statistics.mean(sampled_output)
            self.all_reg_var[design_reg_index[i]] = statistics.variance(sampled_output) / prob.n_init_sol * (1 - prob.n_init_sol / self.n_sol) + statistics.mean(sampled_output_var) / prob.n_init_sol

        return design_reg_index
    
    ##########################################################################################
    def sampled_reg(self):
        ## update intrinsic precision matrix T_e
        # find sampled_regions
        sampled_reg_index = np.argwhere(self.all_reg_var != 0)    # n2 * 1 array
        sampled_reg_index = sampled_reg_index.flatten()    # convert 2d array to 1d array
        return sampled_reg_index

    ##########################################################################################
    def T_epsi_update(self):
        ## update intrinsic precision matrix T_e
        sampled_reg_index = self.sampled_reg()

        # update T_e
        intrinsic_prec_diag = np.zeros(self.n_reg)
        intrinsic_prec_diag[sampled_reg_index] = 1/self.all_reg_var[sampled_reg_index]
        T_epsi = np.diag(intrinsic_prec_diag)

        return T_epsi
    
    ##########################################################################################
    def find_opt(self, dim):
        ## find the region-layer optimum and solution-layer optimum at current iteration

        # find the region-layer optimal node
        opt_val = np.min(self.all_reg_output[np.nonzero(self.all_reg_output)])     # current optimal value
        # print('sampled_reg_output', all_reg_output[np.nonzero(all_reg_output)])
        opt_reg = np.where(self.all_reg_output == opt_val)[0]     # 1 array
        opt_reg = np.take(opt_reg, 0)      # scalar

        # find the current optimal solution and the region-layer node that contains the current optimal solution
        opt_val = np.min(self.all_sol_output[np.nonzero(self.all_sol_output)])
        opt_sol_reg, opt_s = np.where(self.all_sol_output == opt_val)
        opt_sol_reg = np.take(opt_sol_reg, 0)      # 1d array -> scalar
        opt_s = np.take(opt_s, 0)        # 1d array -> scalar

        # opt solution 
        opt_reg_vec = np.unravel_index(opt_sol_reg, self.grid_dim_reg)     # tuple of reg_dim integers
        opt_sol_vec = np.unravel_index(opt_s, self.grid_dim_sol)           # tuple of sol_dim integers

        # get the original solution for recording
        opt_sol_total = np.concatenate((np.asarray(opt_sol_vec), np.asarray(opt_reg_vec))) # concatencate the region and solution layers
        opt_sol_total = opt_sol_total.flatten()   # convert 2d array to 1d array
        orig_opt_sol = np.zeros([1, dim], dtype = np.int32)     # 1 * dim array
        orig_opt_sol[:, self.shift_vector] = opt_sol_total    # use the shift vector to get the original design solution, now [0,10]

        # record the number of sampled solutions in optimal region
        temp = np.argwhere(self.all_sol_output[opt_sol_reg] != 0)    # n2 * 1 array
        temp = temp.flatten()
        num_sampled_in_opt_reg = len(temp)

        return [opt_reg, opt_sol_reg, opt_val, orig_opt_sol[0], num_sampled_in_opt_reg]

    ##########################################################################################
    def region_layer_CEI(self, T_epsi, T_inv, beta, opt_reg):
        ## compute the CEI value at all regions
        ## return the sorted CEI values and the corresponding regions

        # find the sampled regions
        sampled_reg_index = np.argwhere(self.all_reg_output != 0)    # n2 * 1 array
        sampled_reg_index = sampled_reg_index.flatten()    # convert 2d array to 1d array
        # print('sampled_reg_index', sampled_reg_index)

        # conditional means
        cum_change = np.dot(T_epsi, self.all_reg_output - beta)
        reg_cond_mean = beta + np.dot(T_inv[:, sampled_reg_index], cum_change[sampled_reg_index])
        
        # conditional variance and covariance
        reg_cond_var = np.diag(T_inv)    # 1d array of size(n)
        reg_cond_cov = T_inv[:, opt_reg].flatten()    # 1d array of size(n)

        # compute region-layer CEI
        cei_val = CEI(opt_reg, reg_cond_mean, reg_cond_var, reg_cond_cov, -1, -1)
        # find the k region-layer node with largest CEI value (here k is determined by the number of available workers)
        # sort the cei values from largest to smallest and get the corresponding index
        cei_with_index = [(cei_val[i], i) for i in range(self.n_reg)]
        cei_with_index.sort(reverse = True)
        # the first element in sorted_cei_reg_index is the region with largest cei value
        # sorted_cei_reg_index is a tuple, sorted_cei_val will not be used
        sorted_cei_val, sorted_cei_reg_index = zip(*cei_with_index)    
        
        return [sorted_cei_reg_index, sorted_cei_val]
    
    ##########################################################################################
    def reg_output_update(self, reg):
        ##  this function update the region-layer output and variance 
        
        # find the sampled sol in this region 
        sampled_index = np.argwhere(self.all_sol_rep[reg] != 0)    # n_sampled_sol * 1 array
        sampled_index = sampled_index.flatten()   # convert 2d array to 1d array
        
        # compute region-layer output and variance
        n_sampled_sol = len(sampled_index)
        sampled_output = self.all_sol_output[reg, sampled_index]    # n_sampled_sol array
        sampled_output_var = self.all_sol_var[reg, sampled_index]   # n_sampled_sol array
        self.all_reg_output[reg] = statistics.mean(sampled_output)
        self.all_reg_var[reg] = statistics.variance(sampled_output) / n_sampled_sol * (1 - n_sampled_sol / self.n_sol) + statistics.mean(sampled_output_var) / n_sampled_sol

##########################################################################################

class HistoryData:
    def __init__(self, dim):

        # region layer
        self.opt_reg = np.empty((0, 3))     # current optimal region (with smallest region output)
        self.opt_sol_reg = np.empty([])   # region contains the current optimal solution
        self.max_CEI_reg = np.empty([])   # region that has the largest CEI value
        self.reg_CEI_max_val = np.empty([])   # the largest region-layer CEI value
        self.sampled_in_opt_reg = np.empty([])    # the number of sampled solutions in the opt region (5,5,5) index = 665
        self.num_sampled_regions = np.empty([])    # number of sampled regions
        
        # solution layer
        self.opt_sol = np.empty((0, dim), np.int32)  # current optimal solution 
        self.opt_val = np.empty((0, 3))     # current optimal value
        self.opt_true_val = np.empty((0, 3))    # current true value at optimal solution
        self.num_sampled = np.empty([])    # number of sampled solutions
        self.num_reps = np.empty([])      # number of total replications
        self.run_time = np.empty([])      # running time 
        self.total_opt_val = np.array([])       # !!! to check if it's same as cur_opt_true_val
        self.total_max_CEI = np.array([])       # record the CEI value at iterations in the solution layer
        self.cei_thresold_regions = set()
        self.reg_if_repeat = np.empty([])      # record if a region reselected from cei or not
        
        # records in each visited region (reg)

        self.region = np.array([])
        self.max_CEI_in_reg = np.array([])      # the max CEI value in each visited region
        self.sol_opt_in_reg = np.array([])      # the true objectivee function val in each visted region
        self.num_sampled_in_reg = np.array([])     # the number of sampled solutions in each visited region
        self.num_reps_in_reg = np.array([])       # the number of replications ran in each visited region

        self.reg_from_cei = np.empty([])      # record the regions selected from cei value
        self.reg_from_opt = np.empty([])      # record the regions selected from region-layer output
        self.reg_from_min = np.empty([])      # record the regions selected from global optimum

        self.shift_vector = np.array([])  # shift vector at each period
        self.beta = np.array([])     # current GMRF mean hyperparameter
        self.theta = np.array([])   # current single-layer hyperparameters
        self.tau = np.array([])    # current region-layer hyperparameters
    
    ##########################################################################################
    def update_optimum(self, start_time, total_num_visited, true_obj_func, opt_reg, opt_sol_reg, opt_val, orig_opt_sol, num_sampled_in_opt_reg):
        ## update the region-layer and solution-layer optimum

        # -region-layer optimum
        temp_array = np.array([opt_reg, time.time()-start_time, total_num_visited])
        self.opt_reg = np.append(self.opt_reg, temp_array.reshape(1,len(temp_array)), axis = 0)       # record

        # record the current optimal solution and the region-layer node that contains the current optimal solution
        self.opt_sol_reg = np.append(self.opt_sol_reg, opt_sol_reg)
        self.opt_sol = np.append(self.opt_sol, orig_opt_sol.reshape(1,len(orig_opt_sol)), axis = 0)   # current optimal solution 
        temp_array = np.array([opt_val, time.time()-start_time, total_num_visited])
        self.opt_val = np.append(self.opt_val, temp_array.reshape(1,len(temp_array)), axis = 0)     # 1*3 
        temp_array = np.array([true_obj_func(orig_opt_sol - 5), time.time()-start_time, total_num_visited])
        self.opt_true_val = np.append(self.opt_true_val, temp_array.reshape(1,len(temp_array)), axis = 0)    # [-5, 5]

        # print out the current optimum
        # print('current optimal solution', orig_opt_sol)
        # print('current opt: ', opt_val, ', current opt true: ', true_obj_func(orig_opt_sol - 5))

        # record the number of sampled solutions in optimal region
        self.sampled_in_opt_reg = np.append(self.sampled_in_opt_reg, num_sampled_in_opt_reg)

##########################################################################################
##########################################################################################

def master(size, max_time, msg, master_generator, file_name):
    
    workers = range(size - 1)
    status = MPI.Status()

    ########## Initialization ############################################################

    prob = ProblemParameter()
    history = HistoryData(prob.dim)

    ## problem and pGMIA initial setting
    # vector of the no. of feasible solutions in each dimension: [11 11 11 11 11 11]' 
    grid_dim = prob.side * np.ones(prob.dim, dtype = np.int32) 
    n_all = int(np.prod(grid_dim))    # number of all feasible solutions
    
    # projection initialization
    dim_sol = prob.dim - prob.dim_reg   # number of solution-layer dimensions
    shift_vector = projection_selection(prob.dim, prob.dim_reg, master_generator)
    # print('shift vector:', shift_vector)
    data = DataContainer(prob.side, prob.dim, prob.dim_reg, shift_vector)

    ## INITIAL SAMPLES
    design_reg_index = data.pGMIA_initial_sample(prob, master_generator)
    # print('design_reg_index', design_reg_index)

    ## MLE for region and solution layers GMRF hyperparameters
    [beta, tau, theta_sol] = MLE(data.all_reg_output, data.all_reg_var, data.all_sol_output, data.all_sol_var, \
        data.grid_dim_reg, data.grid_dim_sol, design_reg_index, prob.dim_reg, dim_sol, data.n_sol)
    theta_sol[0] = 1/theta_sol[0]

    ## record the GMRF hyperparameter
    theta = np.zeros(prob.dim + 1)
    theta[0] = theta_sol[0]
    # get the theta at solution-layer dimensions
    temp_sol_shift = [shift_vector[i] + 1 for i in range(dim_sol)]
    theta[temp_sol_shift] = theta_sol[1:dim_sol+1]
    # compute the theta at region-layer dimensions
    for i in range(prob.dim_reg):
        theta[shift_vector[dim_sol+i] + 1] = tau[0]*tau[i+1]/(data.n_sol*theta[0])
    # record the GMRF parameters
    # print('shift_vector:', shift_vector)
    history.shift_vector = np.array([shift_vector])  # shift vector at each period
    # later add new shift vector
    # history.shift_vector = np.append(history.shift_vector, np.array([shift_vector]), axis = 0)  # shift vector at each period
    history.beta = np.array([beta])     # current GMRF mean hyperparameter
    history.theta = np.array([theta])   # current single-layer hyperparameters
    history.tau = np.array([tau])    # current region-layer hyperparametes

    ## initialize the region-layer precision matrix T and intrinsic precision matrix T_e
    # print(tau[0])
    # tau[0] = 1e-10
    # tau[1], tau[2], tau[3] = 0.14, 0.14, 0.14
    reg_prec_mat = prec_mat_constr(tau, data.grid_dim_reg)
    T_epsi = data.T_epsi_update()

    print('Initialization finished.')

    ######################################################################################    
    ## send SIMULATE_NEW to all workers

    # set up dataset for master-worker communication
    region_to_worker = dict()   # key: region, value: worker
    worker_to_region = dict()   # key: worker, value: region
    G1 = set()

    # start with iter = 0
    iter = 0

    # record start time
    start_time = time.time()

    # find the region-layer optimum and solution-layer optimum at current iteration
    [opt_reg, opt_sol_reg, opt_val, orig_opt_sol, num_sampled_in_opt_reg] = data.find_opt(prob.dim)
    # update current region- and solution-layer optimum
    total_num_visited = int(np.sum(data.all_sol_rep) / 10) 
    history.update_optimum(start_time, total_num_visited, prob.true_obj_func, opt_reg, opt_sol_reg, opt_val, orig_opt_sol, num_sampled_in_opt_reg)
    # get inverse of region-layer precision matrix
    T = reg_prec_mat + T_epsi
    T_inv = np.linalg.inv(T)

    # compute region-layer CEI at all regions
    [sorted_cei_reg_index, sorted_cei_val] = data.region_layer_CEI(T_epsi, T_inv, beta, opt_reg)

    # simulate the three selected regions, use 'set' to avoid running the same region
    # for reg in set([opt_reg, opt_sol_reg, cei_reg]):
    # reg_to_simulate = set([opt_reg, opt_sol_reg, cei_reg])
    # reg_to_simulate = list(reg_to_simulate)
    reg_to_simulate = set([opt_reg, opt_sol_reg])

    # add margnal CEI regions to simulate sequentially until there are k-1 regions to test
    cei_start = 0
    while len(reg_to_simulate) < size - 1:
        reg_to_simulate.add(sorted_cei_reg_index[cei_start])
        np.append(history.reg_from_cei, sorted_cei_reg_index[cei_start])
        cei_start += 1
    
    # change the set into list
    reg_to_simulate = list(reg_to_simulate)

    # print('region to simulate: ', reg_to_simulate)
    
    rec_time = []
    send_time = []

    # send data to workers
    for w in workers:
        reg = reg_to_simulate[w]

        # sol_rep = data.all_sol_rep[reg]
        # sampled_sol_index = np.argwhere(sol_rep != 0)    # n2 * 1 array
        # sampled_sol_index = sampled_sol_index.flatten()      # convert 2d array to 1d array
        # print('master send to worker in region', reg, 'sampled index', sampled_sol_index)

        # collect all data to send to worker
        send_data = [reg, prob.n_rep, prob.n_init_sol, theta_sol, data.grid_dim_reg, data.grid_dim_sol, shift_vector, prob.obj_func, \
                data.all_sol_rep[reg], data.all_sol_output[reg], data.all_sol_var[reg], start_time, data.count[reg],\
                data.grid_dim, theta, [], -1, -1, -1, -1]
        comm.send(send_data, dest = w, tag = tags.SIMULATE_NEW)
        # print('Master send SIMULATE_NEW to worker ', w, ' in region ', reg, ' AT time:', time.time() - start_time)

        # update master-worker communication dataset
        region_to_worker[reg] = w   # key: region, value: worker
        worker_to_region[w] = reg   # key: worker, value: region
        cur_x_opt_output = np.inf
        # G1 = set()   no change at this time
    send_time.append(time.time()-start_time)

    ######################################################################################
    ## start pGMIA with fixed budget in simulation time 
    while time.time() - start_time < max_time:  ## !!!!! time constraint as stopping criterion

        # print('########## iter', iter)
        # receive message from a worker
        rec_data = comm.recv(source = MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status = status)
        rec_time.append(time.time()-start_time)
        worker_id = status.Get_source()
        tag = status.Get_tag()
        reg = rec_data[-1]
        if rec_data[-3] < delta:
            history.cei_thresold_regions.add(reg)
        # print('Master receive data from worker', worker_id, 'with messsage ', msg[tag])

        # check number of sampled regions
        sampled_reg_index = data.sampled_reg()
        # print('Current number of sampled regions: ', len(sampled_reg_index))
        # check number of samples
        num_sampled = int(np.count_nonzero(data.all_sol_rep))
        num_reps = int(np.sum(data.all_sol_rep))
        # print('Current number of sampled solutions: ', num_sampled, ' with ', num_reps, ' replications')
        beta = likelihood_new(tau[0], tau[1:4], data.grid_dim_reg, sampled_reg_index, data.all_reg_output[sampled_reg_index], np.diag(data.all_reg_var[sampled_reg_index]), 0, 0)[1]
        # print('New beta: ', beta)

        # If the worker finishes its job or the master want to finish its job
        if tag == tags.FINISH or reg in G1:
            [sol_rep, sol_output, sol_var, cur_opt_true_val, cur_max_CEI, \
                cur_posterior_mean, cur_posterior_var, \
                x_opt, x_opt_output, x_opt_output_var, x_opt_cond_mean, x_opt_cond_var, \
                max_CEI_val, count, reg]  = rec_data
            data.count[reg] = count
            # udpate new-simulated data within region
            data.all_sol_rep[reg], data.all_sol_output[reg], data.all_sol_var[reg] = sol_rep, sol_output, sol_var
            data.reg_output_update(reg)

            # update global optimum
            if x_opt_output < cur_x_opt_output:
                cur_x_opt = x_opt
                cur_x_opt_output = x_opt_output  
                cur_x_opt_output_var = x_opt_output_var
                cur_x_opt_cond_mean = x_opt_cond_mean
                cur_x_opt_cond_var = x_opt_cond_var

            # update master-worker communication dataset
            del region_to_worker[reg]   # key: region, value: worker
            del worker_to_region[worker_id]  # key: worker, value: region
            if reg in G1: 
                G1.remove(reg)
            # G1 = set()   no change at this time

            # find the region-layer optimum and solution-layer optimum at current iteration
            [opt_reg, opt_sol_reg, opt_val, orig_opt_sol, num_sampled_in_opt_reg] = data.find_opt(prob.dim)
            # update current region- and solution-layer optimum
            total_num_visited = int(np.sum(data.all_sol_rep) / 10) 
            history.update_optimum(start_time, total_num_visited, prob.true_obj_func, opt_reg, opt_sol_reg, opt_val, orig_opt_sol, num_sampled_in_opt_reg)

            # record number of sampled solutions and replications
            history.num_sampled = np.append(history.num_sampled, int(np.count_nonzero(data.all_sol_rep)))
            history.num_reps = np.append(history.num_reps, int(np.sum(data.all_sol_rep)))

            # # open a file, where you ant to store the data
            # file = open(file_name, 'wb')
            # # file = open('asyn_seed12', 'wb')
            # # dump information to that file
            # pickle.dump(history, file)
            # # close the file
            # file.close()

            # if the projection update criterion is satisfied
            if iter % prob.projection_update_period == prob.projection_update_period-1:

                # print('#################################### Update the projection')

                # send QUIT_JOB to all workers 
                for w in workers:
                    send_data = [w]
                    comm.send(send_data, dest = w, tag = tags.QUIT_JOB)
                    # print('Master send QUIT_JOB to worker ', w)
                send_time.append(time.time()-start_time)

                # receive message from all workers
                for bbb in range(len(workers) - 1):
                    rec_data = comm.recv(source = MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status = status)
                    if bbb == 0:
                        rec_time.append(time.time()-start_time)
                        # print('rec time: ', time.time()-start_time, 'len', len(rec_time))
                    worker_id = status.Get_source()
                    # print('Master receive data from worker', worker_id)
                    [sol_rep, sol_output, sol_var, cur_opt_true_val, cur_max_CEI, \
                        cur_posterior_mean, cur_posterior_var, \
                        x_opt, x_opt_output, x_opt_output_var, x_opt_cond_mean, x_opt_cond_var, \
                        max_CEI_val, count, reg]  = rec_data
                    data.count[reg] = count
                    # udpate new-simulated data within region
                    data.all_sol_rep[reg], data.all_sol_output[reg], data.all_sol_var[reg] = sol_rep, sol_output, sol_var
                    data.reg_output_update(reg)

                    # update master-worker communication dataset
                    del region_to_worker[reg]   # key: region, value: worker
                    del worker_to_region[worker_id]  # key: worker, value: region
                

                # find all sampled solutions and transfer it into n2 * dim array
                sampled_temp = np.nonzero(data.all_sol_rep)
                n_sampled = len(sampled_temp[0])     # number of sampled solutions
                reg_vec_temp = np.unravel_index(sampled_temp[0], data.grid_dim_reg)   # tuple of vectors at region layer 
                sol_vec_temp = np.unravel_index(sampled_temp[1], data.grid_dim_sol)   # tuple of vectors at solution layer
                shift_sampled_vec = np.transpose(np.concatenate((sol_vec_temp, reg_vec_temp)))    # n2 * dim array of vectors after shifting
                
                # create an empty n2 * dim array
                sampled_vec = np.zeros([n_sampled, data.dim], dtype = np.int32)
                sampled_vec[:, shift_vector] = shift_sampled_vec     # n2 * dim array, each row represents a sampled solution

                # sampled output and sampled variance
                sampled_rep = data.all_sol_rep[sampled_temp]          # 1d array, size = n2
                sampled_output = data.all_sol_output[sampled_temp]      # 1d array, size = n2
                sampled_var = data.all_sol_var[sampled_temp]        # 1d array, size = n2

                # select a projection
                shift_vector = projection_selection(prob.dim, prob.dim_reg, master_generator)
                # print('shift vector:', shift_vector)
                data = DataContainer(prob.side, prob.dim, prob.dim_reg, shift_vector)

                # return the parameters and updated dataset
                output = projection_data_update(prob.n_init_reg, prob.n_init_sol, prob.n_rep, data.grid_dim, prob.dim_reg, \
                    prob.obj_func, shift_vector, sampled_vec, sampled_rep, sampled_output, sampled_var)
                # update the dataset in the new projection
                [theta, beta, tau, theta_sol, reg_prec_mat, \
                    data.all_sol_rep, data.all_sol_output, data.all_sol_var, data.all_reg_output, data.all_reg_var] = output

                # later add new shift vector
                history.shift_vector = np.append(history.shift_vector, np.array(shift_vector).reshape(1,len(shift_vector)), axis = 0)  # shift vector at each period
                # history.shift_vector = np.append(history.shift_vector, np.array([shift_vector]), axis = 0)  # shift vector at each period
                history.beta = np.append(history.beta, beta)     # current GMRF mean hyperparameter
                history.theta = np.append(history.theta, theta)   # current single-layer hyperparameters
                history.tau = np.append(history.tau, tau)    # current region-layer hyperparametes

                ## initialize the region-layer precision matrix T and intrinsic precision matrix T_e
                reg_prec_mat = prec_mat_constr(tau, data.grid_dim_reg)
                T_epsi = data.T_epsi_update()

                ## send SIMULATE_NEW to all workers

                # set up dataset for master-worker communication
                region_to_worker = dict()   # key: region, value: worker
                worker_to_region = dict()   # key: worker, value: region
                G1 = set()
                cei_thresold_regions = set()

                # find the region-layer optimum and solution-layer optimum at current iteration
                [opt_reg, opt_sol_reg, opt_val, orig_opt_sol, num_sampled_in_opt_reg] = data.find_opt(prob.dim)
                # update current region- and solution-layer optimum
                total_num_visited = int(np.sum(data.all_sol_rep) / 10) 
                history.update_optimum(start_time, total_num_visited, prob.true_obj_func, opt_reg, opt_sol_reg, opt_val, orig_opt_sol, num_sampled_in_opt_reg)
                # get inverse of region-layer precision matrix
                T = reg_prec_mat + T_epsi
                T_inv = np.linalg.inv(T)

                # compute region-layer CEI at all regions
                [sorted_cei_reg_index, sorted_cei_val] = data.region_layer_CEI(T_epsi, T_inv, beta, opt_reg)

                # simulate the three selected regions, use 'set' to avoid running the same region
                reg_to_simulate = set([opt_reg, opt_sol_reg])

                # add margnal CEI regions to simulate sequentially until there are k-1 regions to test
                cei_start = 0
                while len(reg_to_simulate) < size - 1:
                    reg_to_simulate.add(sorted_cei_reg_index[cei_start])
                    cei_start += 1
                
                # change the set into list
                reg_to_simulate = list(reg_to_simulate)

                # print('region to simulate: ', reg_to_simulate)

                # send data to workers
                for w in workers:
                    reg = reg_to_simulate[w]

                    # collect all data to send to worker
                    send_data = [reg, prob.n_rep, prob.n_init_sol, theta_sol, data.grid_dim_reg, data.grid_dim_sol, shift_vector, prob.obj_func, \
                        data.all_sol_rep[reg], data.all_sol_output[reg], data.all_sol_var[reg], start_time, data.count[reg],\
                        data.grid_dim, theta, [], -1, -1, -1, -1]
                    comm.send(send_data, dest = w, tag = tags.SIMULATE_NEW)
                    # print('Master send SIMULATE_NEW to worker ', w, ' in region ', reg, ' AT time:', time.time() - start_time)

                    # update master-worker communication dataset
                    region_to_worker[reg] = w   # key: region, value: worker
                    worker_to_region[w] = reg   # key: worker, value: region
                    cur_x_opt_output = np.inf
                send_time.append(time.time()-start_time)
                # print('send time: ', time.time()-start_time, 'len', len(send_time))

                # # open a file, where you ant to store the data
                # file = open(file_name, 'wb')
                # # file = open('asyn_seed12', 'wb')
                # # dump information to that file
                # pickle.dump(history, file)
                # # close the file
                # file.close()


            # if tulde R changes, we sent "QUIT_JOB" to all workers, then wait until all regions send back and send new (### CASE 1)
            elif opt_reg != history.opt_reg[-2][0]:     # it is -2 becuase the -1 is just opt_reg and we just update it
                
                # print('#################################### Current tulde R changes')

                # send QUIT_JOB to all workers 
                for w in workers:
                    send_data = [w]
                    comm.send(send_data, dest = w, tag = tags.QUIT_JOB)
                    # print('Master send QUIT_JOB to worker ', w)
                send_time.append(time.time()-start_time)
                
                # receive message from all workers
                for bbb in range(len(workers) - 1):
                    rec_data = comm.recv(source = MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status = status)
                    if bbb == 0:
                        rec_time.append(time.time()-start_time)
                    worker_id = status.Get_source()
                    # print('Master receive data from worker', worker_id)
                    [sol_rep, sol_output, sol_var, cur_opt_true_val, cur_max_CEI, \
                        cur_posterior_mean, cur_posterior_var, \
                        x_opt, x_opt_output, x_opt_output_var, x_opt_cond_mean, x_opt_cond_var, \
                        max_CEI_val, count, reg]  = rec_data
                    # udpate new-simulated data within region
                    data.count[reg] = count
                    data.all_sol_rep[reg], data.all_sol_output[reg], data.all_sol_var[reg] = sol_rep, sol_output, sol_var
                    data.reg_output_update(reg)

                    # update master-worker communication dataset
                    del region_to_worker[reg]   # key: region, value: worker
                    del worker_to_region[worker_id]  # key: worker, value: region

                # update global optimum
                if x_opt_output < cur_x_opt_output:
                    cur_x_opt = x_opt
                    cur_x_opt_output = x_opt_output  
                    cur_x_opt_output_var = x_opt_output_var
                    cur_x_opt_cond_mean = x_opt_cond_mean
                    cur_x_opt_cond_var = x_opt_cond_var

                # find the region-layer optimum and solution-layer optimum at current iteration
                [opt_reg, opt_sol_reg, opt_val, orig_opt_sol, num_sampled_in_opt_reg] = data.find_opt(prob.dim)
                # update current region- and solution-layer optimum
                total_num_visited = int(np.sum(data.all_sol_rep) / 10) 
                history.update_optimum(start_time, total_num_visited, prob.true_obj_func, opt_reg, opt_sol_reg, opt_val, orig_opt_sol, num_sampled_in_opt_reg)


                # update T_epsi
                T_epsi = data.T_epsi_update()

                # get inverse of region-layer precision matrix
                T = reg_prec_mat + T_epsi
                T_inv = np.linalg.inv(T)

                # compute region-layer CEI at all regions
                [sorted_cei_reg_index, sorted_cei_val] = data.region_layer_CEI(T_epsi, T_inv, beta, opt_reg)
                # print('region CEI values: ', sorted_cei_val[0:5])

                # simulate the three selected regions, use 'set' to avoid running the same region
                # for reg in set([opt_reg, opt_sol_reg, cei_reg]):
                # reg_to_simulate = set([opt_reg, opt_sol_reg, cei_reg])
                # reg_to_simulate = list(reg_to_simulate)
                reg_to_simulate = set([opt_reg, opt_sol_reg])

                # add margnal CEI regions to simulate sequentially until there are k-1 regions to test
                cei_start = 0
                while len(reg_to_simulate) < size - 1:
                    reg_to_simulate.add(sorted_cei_reg_index[cei_start])
                    cei_start += 1
                
                # change the set into list
                reg_to_simulate = list(reg_to_simulate)

                # print('region to simulate: ', reg_to_simulate)

                # send data to workers
                for w in workers:
                    reg = reg_to_simulate[w]

                    # collect all data to send to worker
                    send_data = [reg, prob.n_rep, prob.n_init_sol, theta_sol, data.grid_dim_reg, data.grid_dim_sol, shift_vector, prob.obj_func, \
                        data.all_sol_rep[reg], data.all_sol_output[reg], data.all_sol_var[reg], start_time, data.count[reg],\
                        data.grid_dim, theta, cur_x_opt, cur_x_opt_output, cur_x_opt_output_var, cur_x_opt_cond_mean, cur_x_opt_cond_var]
                    comm.send(send_data, dest = w, tag = tags.SIMULATE_NEW)
                    # print('Master send SIMULATE_NEW to worker ', w, ' in region ', reg, ' AT time:', time.time() - start_time)

                    # update master-worker communication dataset
                    region_to_worker[reg] = w   # key: region, value: worker
                    worker_to_region[w] = reg   # key: worker, value: region
                    # G1 = set()   no change at this time
                send_time.append(time.time()-start_time)
            
            # if R min changes, we add previous R min into G1, and then send SIMULATE_NEW to cur R min (### CASE 2)
            elif opt_sol_reg != history.opt_sol_reg[-2] and opt_sol_reg not in region_to_worker:     
                
                # print('#################################### Current R min changes to', opt_sol_reg, ' from ', history.opt_sol_reg[-2])
                # add previous R min into G1
                G1.add(history.opt_sol_reg[-2])

                # collect all data to send to worker
                reg = opt_sol_reg
                send_data = [reg, prob.n_rep, prob.n_init_sol, theta_sol, data.grid_dim_reg, data.grid_dim_sol, shift_vector, prob.obj_func, \
                        data.all_sol_rep[reg], data.all_sol_output[reg], data.all_sol_var[reg], start_time, data.count[reg],\
                        data.grid_dim, theta, cur_x_opt, cur_x_opt_output, cur_x_opt_output_var, cur_x_opt_cond_mean, cur_x_opt_cond_var]
                comm.send(send_data, dest = worker_id, tag = tags.SIMULATE_NEW)
                send_time.append(time.time()-start_time)
                # print('Master send SIMULATE_NEW to worker ', worker_id, ' in region ', reg, ' at time:', time.time() - start_time)

                # update master-worker communication dataset
                region_to_worker[reg] = worker_id   # key: region, value: worker
                worker_to_region[worker_id] = reg   # key: worker, value: region
                # G1 = set()   no change at this time
                

            
            # both tulde R and R min don't change, just find R cei, SIMULATE NEW (### CASE 3)
            else:
                
                # update T_epsi
                T_epsi = data.T_epsi_update()

                # get inverse of region-layer precision matrix
                T = reg_prec_mat + T_epsi
                T_inv = np.linalg.inv(T)

                # compute region-layer CEI at all regions
                [sorted_cei_reg_index, sorted_cei_val] = data.region_layer_CEI(T_epsi, T_inv, beta, opt_reg)
                # print('region CEI values: ', sorted_cei_val[0:5])

                # record
                history.max_CEI_reg = np.append(history.max_CEI_reg, sorted_cei_reg_index[0])     # region that has the largest CEI value, scalar
                history.reg_CEI_max_val = np.append(history.reg_CEI_max_val, sorted_cei_val[0])   # scalar

                # record number of sampled solutions and replications
                history.num_sampled = np.append(history.num_sampled, int(np.count_nonzero(data.all_sol_rep)))
                history.num_reps = np.append(history.num_reps, int(np.sum(data.all_sol_rep)))

                # choose the idle region with largest marginal CEI value
                # then add margnal CEI regions to simulate
                cei_start = 0
                while sorted_cei_reg_index[cei_start] in region_to_worker:
                    cei_start += 1
                
                # the previous while loop stops when sorted_cei_reg_index[cei_start] not in region_to_worker
                reg = sorted_cei_reg_index[cei_start]
                
                if reg in history.reg_CEI_max_val:
                    history.reg_if_repeat = np.append(history.reg_if_repeat, int(0))
                else:
                    history.reg_if_repeat = np.append(history.reg_if_repeat, int(1))


                # collect all data to send to worker
                send_data = [reg, prob.n_rep, prob.n_init_sol, theta_sol, data.grid_dim_reg, data.grid_dim_sol, shift_vector, prob.obj_func, \
                        data.all_sol_rep[reg], data.all_sol_output[reg], data.all_sol_var[reg], start_time, data.count[reg],\
                        data.grid_dim, theta, cur_x_opt, cur_x_opt_output, cur_x_opt_output_var, cur_x_opt_cond_mean, cur_x_opt_cond_var]
                comm.send(send_data, dest = worker_id, tag = tags.SIMULATE_NEW)
                send_time.append(time.time()-start_time)
                # print('Master send SIMULATE_NEW to worker ', worker_id, ' in region ', reg, ' at time:', time.time() - start_time)

                # update master-worker communication dataset
                region_to_worker[reg] = worker_id   # key: region, value: worker
                worker_to_region[worker_id] = reg   # key: worker, value: region
                # G1 = set()   no change at this time

            iter += 1
            
        
        # if receiving "UPDATE" from the worker, we just let it continue
        else:     # (### CASE 4)

            # collect all data to send to worker
            send_data = []
            comm.send(send_data, dest = worker_id, tag = tags.CONTINUE_OLD)
            send_time.append(time.time()-start_time)
            # print('Master send CONTINUE_OLD to worker ', worker_id, ' in region ', reg, ' at time:', time.time() - start_time)


    # receive message from all workers   (### CASE 5)
    for bbb in workers:
        rec_data = comm.recv(source = MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status = status)
        if bbb == 0:
            rec_time.append(time.time()-start_time)
        worker_id = status.Get_source()
        # print('Master receive data from worker', worker_id)
        [sol_rep, sol_output, sol_var, cur_opt_true_val, cur_max_CEI, \
                cur_posterior_mean, cur_posterior_var, \
                x_opt, x_opt_output, x_opt_output_var, x_opt_cond_mean, x_opt_cond_var, \
                max_CEI_val, count, reg]  = rec_data
        data.count[reg] = count
        data.all_sol_rep[reg], data.all_sol_output[reg], data.all_sol_var[reg] = sol_rep, sol_output, sol_var
        data.reg_output_update(reg)

        # update master-worker communication dataset
        del region_to_worker[reg]   # key: region, value: worker
        del worker_to_region[worker_id]  # key: worker, value: region

    # send TERMINATE_ALG to all workers 
    for w in workers:
        send_data = [w]
        comm.send(send_data, dest = w, tag = tags.TERMINATE_ALG)
        # print('Master send TERMINATE_ALG to worker ', w)
    

    return history

##########################################################################################

def worker(worker_id, msg, delta, update_iter, max_number_udpates, worker_generator):

    rec_time = []
    send_time = []

    while True:
        
        # receive message from master
        rec_data = comm.recv(source = master_rank, tag = MPI.ANY_TAG, status = status)
        start_cur = time.time()
        tag = status.Get_tag()
        # print('Worker ', worker_id, ' receive data from master, the message is: ', msg[tag])

        # if receive SIMULATE_NEW, get the details of received data
        if tag == tags.SIMULATE_NEW:
            [this_reg, n_rep, n_init_sol, theta_sol, grid_dim_reg, grid_dim_sol, shift_vector, obj_func, \
                sol_rep, sol_output, sol_var, start_time, count,\
                grid_dim, theta, cur_x_opt, cur_x_opt_output, cur_x_opt_output_var, cur_x_opt_cond_mean, cur_x_opt_cond_var] = rec_data
            number_updates = 0
            
        elif tag == tags.CONTINUE_OLD:
            count += 1
        
        # if receive QUIT_JOB, continue
        elif tag == tags.QUIT_JOB:
            # print('Worker ', worker_id, ' continue to next loop')
            continue
        
        # if receive TERMINATE_ALG, break
        elif tag == tags.TERMINATE_ALG:
            # print('Worker ', worker_id, ' break the algorithm')
            break
        
        rec_time.append(start_cur - start_time)
        # When receiving SIMULATE_NEW or CONTINUE_OLD, do solution-layer exploration    
        # for each region, run solution-layer GMIA
        # output = [sol_rep, sol_output, sol_var, cur_opt_true_val, cur_max_CEI, max_CEI_val]
        output = FineLevel_GMRF(this_reg, n_rep, n_init_sol, theta_sol, grid_dim_reg, grid_dim_sol, shift_vector, obj_func, \
                sol_rep, sol_output, sol_var, update_iter, delta, count, worker_generator,\
                grid_dim, theta, cur_x_opt, cur_x_opt_output, cur_x_opt_output_var, cur_x_opt_cond_mean, cur_x_opt_cond_var)
        max_CEI_val = output[-2]
        # print('max_CEI_val', max_CEI_val, ' with in region ', this_reg)
        count_at_this_reg = output[-1]

        # send data to the master
        # output = [sol_rep, sol_output, sol_var, cur_opt_true_val, cur_max_CEI, max_CEI_val, this_reg]
        output.append(this_reg)

        # if stopping condition is satisfied, send "FINISH" to master
        # print('max_CEI_val', max_CEI_val, 'count', number_updates)
        if max_CEI_val < delta or number_updates == max_number_udpates:    # count
            send_data = output
            comm.send(send_data, dest = master_rank, tag = tags.FINISH)
            # print('Worker ', worker_id, ' send FINISH to master at time', time.time() - start_time)

        # if updating condition is satisfied, send "UPDATE" to master
        else:
            send_data = output
            comm.send(send_data, dest = master_rank, tag = tags.UPDATE)
            # print('Worker ', worker_id, ' send UPDATE to master at time', time.time() - start_time)
            number_updates += 1

        send_time.append(time.time()-start_time)
    
    return [rec_time, send_time]

##########################################################################################
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

num_workers = size - 1
master_rank = size - 1

# print(sys.argv)
seed = int(sys.argv[1])
file_name = 'seed' + str(sys.argv[1]) + 'c' + str(size) + 'all'

# seed = 12
seed_on_this_processor = size * seed + rank
bit_generator = np.random.MT19937(seed_on_this_processor)
processor_generator = np.random.Generator(bit_generator)

msg = ["SIMULATE_NEW", "CONTINUE_OLD", "QUIT_JOB", "TERMINATE_ALG", "FINISH", "UPDATE"]

##### MASTER
if rank == master_rank:
    max_time = 1200
    print('----------- START ALGORITHM ------------------------')
    delta = 1e-2
    output = master(size, max_time, msg, processor_generator, file_name)

    # open a file, where you ant to store the data
    file = open(file_name, 'wb')
    # file = open('asyn_seed12', 'wb')
    # dump information to that file
    pickle.dump(output, file)
    # close the file
    file.close()

##### WORKER
else:
    status = MPI.Status()
    # print('status', status)
    delta = 1e-2
    update_iter = 5
    max_number_udpates = 4
    t = worker(rank, msg, delta, update_iter, max_number_udpates, processor_generator)
    
# mpiexec -n 8 python -u main_asyn.py 1 'seed1'
# mpiexec -n 8 python main_asyn.py