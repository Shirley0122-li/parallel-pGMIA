from mpi4py import MPI
import numpy as np
import time
import statistics

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    n_reg = 11 ** 3
    n_sol = 11 ** 3

    master_receive = [[0, 0] for _ in range(size)]
    master_send = [[0, 0] for _ in range(size)]
    worker_receive = [[0, 0] for _ in range(size)]
    worker_send = [[0, 0] for _ in range(size)]

    # [4,5,6,7]  problem size
    # shift vector [1,2,3,4]
    # n_reg 6*7  ,n_sol 4*5
    # 

    # shift_vector [3,1,4,2]
    # n_reg 7*5  n_sol 4*6

    # single-layer -> all solutions
    # [i,j] of the following matrices represents the jth solution at ith region
    # [i,j]
    all_sol_rep = np.random.random_sample([n_reg, n_sol])    # number of replications at each solution
    all_sol_output = np.random.random_sample([n_reg, n_sol])   # output mean at each solution
    all_sol_var = np.random.random_sample([n_reg, n_sol])    # output variance at each solution

    # region-layer -> all region-layer nodes
    # the [i] of the following vector represents the ith region
    all_reg_output = np.empty(n_reg)  # output mean at each region
    all_reg_output[:] = np.nan
    all_reg_var = np.empty(n_reg)   # output variance at each region
    all_reg_var[:] = np.nan

    shift_vector = [5,3,4,2,1,0]

    # send data to workers
    for i in range(size-1):
        master_send[i][0] = time.time()
        data = [all_sol_rep[i], all_sol_output[i], all_sol_var[i], shift_vector]
        req = comm.isend(data, dest=i + 1, tag=11)
        req.wait()
        master_send[i][1] = time.time()
        print('send data to worker', i + 1, 'time:', master_send[i][1] - master_send[i][0])

    # receive data from workers
    for i in range(size-1):
        master_receive[i][0] = time.time()
        req = comm.irecv(source = i + 1, tag=11)
        temp = req.wait()
        all_sol_rep[i+1] = temp[0]
        all_sol_output[i+1] = temp[1]
        all_sol_var[i+1] = temp[2]
        master_receive[i][1] = time.time()
        print('receive data from worker', temp[-1], 'time:', master_receive[i][1] - master_receive[i][0])
        all_reg_output[i+1] = statistics.mean(temp[1])
        all_reg_var[i+1] = statistics.mean(temp[2])



elif rank > 0:

    master_receive = [[0, 0] for _ in range(size)]
    master_send = [[0, 0] for _ in range(size)]
    worker_receive = [[0, 0] for _ in range(size)]
    worker_send = [[0, 0] for _ in range(size)]

    # receive data from the master
    i = rank - 1
    worker_receive[i][0] = time.time()
    req = comm.irecv(source=0, tag=11)
    data = req.wait()
    worker_receive[i][1] = time.time()
    print('worker', rank, 'receive data from master', 'time:', worker_receive[i][1] - worker_receive[i][0])

    sol_rep = data[0]
    sol_output = data[1]
    sol_var = data[2]

    sol_rep[rank] = rank
    sol_output[rank] = rank + 50
    sol_var[rank] = rank ** 2

    # send data to the master
    worker_send[i][0] = time.time()
    data = [sol_rep, sol_output, sol_var, rank]
    req = comm.isend(data, dest=0, tag=11)
    req.wait()
    worker_send[i][1] = time.time()
    print('worker', rank, 'send data to master', 'time:', worker_send[i][1] - worker_send[i][0])


# mpiexec -n 8 python example.py