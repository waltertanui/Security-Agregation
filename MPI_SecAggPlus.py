import pickle as pickle
import random
import sys
import time

import numpy as np
from mpi4py import MPI

from sec_agg.mpc_function import BGW_decoding, my_pk_gen, BGW_encoding
from sec_agg.common_function import get_group_idxset

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if len(sys.argv) == 1:
    if rank == 0:
        print("ERROR: please input the number of workers")
    exit()
elif len(sys.argv) == 2:
    N = int(sys.argv[1])
    d = 100080 # 100*1000    # 100k
    is_sleep = False
elif len(sys.argv) == 3:
    N = int(sys.argv[1])
    d = int(sys.argv[2])
    is_sleep = False
elif len(sys.argv) == 4:
    N = int(sys.argv[1])
    d = int(sys.argv[2])
    is_sleep = True
    comm_mbps = float(sys.argv[3]) # unit: Mbps
else:
    if rank ==0:
        print("ERROR: please check the input arguments")
    exit()

T = int(N / 2)

# set system parameters
p = 2 ** 31 - 1  # p = 2 ** 15 - 19
g = 13  # generator for the key agreement. It should be a prime number.
n_trial = 5  # average 100 trials

# my_model = np.random.randint(0,p,size=(d))
my_model = np.ones(shape=(d)).astype(int)

num_pk_per_user = 2

drop_rate_array = np.array([0.1, 0.3, 0.5])
time_out = []

for t in range(len(drop_rate_array)):
    drop_rate = drop_rate_array[t]
    time_avg = np.zeros((4), dtype=float)

    if rank == 0:
        print('#############################################################################')
        print('                         drop_rate=', drop_rate, 'starts!!')
        print('\n')
    for avg_idx in range(n_trial):
        comm.Barrier()
        ##########################################
        ##          Server starts HERE          ##
        ##########################################
        if rank == 0:  # Server
            # Round 0: Advertise Keys
            comm.Barrier()

            # 0.0. Receive the public keys
            t0 = time.time()

            public_key_list = np.empty(shape=(num_pk_per_user, N), dtype='int64')
            for i in range(N):
                # drop_info = np.empty(N, dtype='int')
                data = np.empty(num_pk_per_user, dtype='int64')
                comm.Recv(data, source=i + 1)
                public_key_list[:, i] = data

            # 0.1.Broadcast the public keys
            for i in range(N):
                data = np.reshape(public_key_list, num_pk_per_user * N)
                comm.Send(data, dest=i + 1)

            comm.Barrier()
            # print '[ rank= ',rank,']',public_key_list
            t0 = time.time() - t0

            # Round 1. Share Key

            # 1.0. Receive the SS from users
            t1 = time.time()

            b_u_SS_list = np.empty((N, N), dtype='int64')
            s_sk_SS_list = np.empty((N, N), dtype='int64')

            # print np.shape(b_u_SS_list)

            for i in range(N):
                data = np.empty(N, dtype='int64')
                comm.Recv(data, source=i + 1)
                b_u_SS_list[i, :] = data

                data = np.empty(N, dtype='int64')
                comm.Recv(data, source=i + 1)
                s_sk_SS_list[i, :] = data

            # 1.1. Send the SS to the users

            for i in range(N):
                data = b_u_SS_list[:, i].astype('int64')
                comm.Send(data, dest=i + 1)

                data = s_sk_SS_list[:, i].astype('int64')
                comm.Send(data, dest=i + 1)

            comm.Barrier()
            t1 = time.time() - t1

            # Round 2. Masked Input Collection
            t2 = time.time()

            np.random.seed(t * n_trial + avg_idx)

            drop_info = np.zeros(N, dtype='int')
            n_users_drop = np.round(drop_rate * N).astype(int)
            dropuser_sel = random.sample(range(N), n_users_drop)
            drop_info[dropuser_sel] = 1

            # drop_info = np.random.binomial(size=N, n=1, p = drop_rate).astype(int)
            # drop_info[0] = 1

            data_rv = np.zeros((N, d), dtype='int')
            for i in range(N):
                comm.Recv(data_rv[i, :], source=i + 1)

            z = np.zeros(d, dtype='int')
            for i in range(N):
                if drop_info[i] == 0:
                    z = np.mod(z + data_rv[i, :], p)

            # print '[ rank= ',rank,'] bf unmasking: z=',z
            comm.Barrier()
            t2 = time.time() - t2

            # Round 3. Unmasking
            t3 = time.time()

            # considering worst scenario of dropout

            # 3.0. Send drop_info 
            for i in range(N):
                comm.Send(drop_info, dest=i + 1)

            # 3.1. Receive SS (b_u_SS for surviving users, s_sk_SS for dropped users)
            SS_rx = np.empty((N, N), dtype='int64')
            for i in range(N):
                data = np.empty(N, dtype='int64')
                comm.Recv(data, source=i + 1)
                SS_rx[:, i] = data

            # print SS_rx
            # print b_u_SS_list

            # 3.2. Generate PRG based on the seed

            for i in range(N):
                if drop_info[i] == 0:
                    SS_input = np.reshape(SS_rx[i, 0:T + 1], (T + 1, 1))
                    # SS_input = np.reshape(b_u_SS_list[i,0:T+1],(T+1,1))
                    b_u = BGW_decoding(SS_input, range(T + 1), p)
                    np.random.seed(b_u[0][0])
                    temp = np.random.randint(0, p, size=d).astype(int)
                    # temp = np.random.randint(0, p, size=d, dtype='int')
                    z = np.mod(z - temp, p)
                    # print i, b_u, temp
                else:
                    mask = np.zeros(d, dtype='int')
                    SS_input = np.reshape(SS_rx[i, 0:T + 1], (T + 1, 1))
                    s_sk_dec = BGW_decoding(SS_input, range(T + 1), p)

                    my_group_idx = get_group_idxset(N,i)

                    for j in my_group_idx:
                        if j != 1:
                            s_pk_list_ = public_key_list[1,:]
                            s_uv_dec = np.mod(s_sk_dec[0][0] * s_pk_list_[j], p)

                            if j == i:
                                temp = np.zeros(d,dtype='int')
                            elif j < i:
                                #SS_input = np.reshape(b_u_SS_list[i,0:T+1],(T+1,1))
                                #s_sk = BGW_decoding(SS_input,range(T+1),p)
                                #np.random.seed(s_sk[0][0])
                                #temp = np.random.randint(0, p, size=d, dtype='int')
                                #np.random.seed(s_uv[j-1])
                                np.random.seed(s_uv_dec)
                                temp = np.random.randint(0, p, size=d).astype(int)
                            else:
                                #np.random.seed(s_uv[j-1])
                                np.random.seed(s_uv_dec)
                                temp = -np.random.randint(0, p, size=d).astype(int)
                            #print 'seed, temp=',s_uv_dec,temp
                            mask = np.mod(mask+temp,p)

                    # print 'mask =', mask
                    z = np.mod(z + mask, p)

                    # print '[ rank= ',rank,'] af unmasking: z=',z,'\n'
            comm.Barrier()
            t3 = time.time() - t3

            time_set = np.array([t0, t1, t2, t3])

            # print avg_idx, '-th trial time info=', time_set
            time_avg += time_set

            print(avg_idx, '-th trial, # drop users =', np.sum(drop_info))
            print(avg_idx, '-th trial, time=', time_set)

        ##########################################
        ##           Users start HERE           ##
        ##########################################
        elif rank <= N:  # Users

            # Round 0. Advertise Keys
            comm.Barrier()

            # 0.0. Send my public keys
            np.random.seed(rank)
            my_sk = np.random.randint(0, p, size=(2)).astype('int64')
            my_pk = my_pk_gen(my_sk, p, 0)

            my_key = np.concatenate((my_pk, my_sk))  # length=4 : c_pk, s_pk, c_sk, s_sk

            comm.Send(my_key[0:2], dest=0)  # send public key to the server

            # print '[ rank= ',rank,']', my_key[0:2]

            # 0.1. Rx public key list from the server
            public_key_list = np.empty(num_pk_per_user * N).astype('int64')
            comm.Recv(public_key_list, source=0)

            public_key_list = np.reshape(public_key_list, (num_pk_per_user, N))

            # print '[ rank= ',rank,']', public_key_list

            # Round 1. Share Key
            comm.Barrier()
            # 1.1 generate b_u, s_uv

            s_pk_list = public_key_list[1, :]
            my_s_sk = my_key[3]
            my_c_sk = my_key[2]

            b_u = my_c_sk
            s_uv = np.mod(s_pk_list * my_s_sk, p)

            # print '[ rank= ',rank,']', s_uv

            # 1.2. generate SS of b_u, s_sk
            SS_input = np.reshape(np.array([my_c_sk, my_s_sk]), (2, 1))
            my_SS = BGW_encoding(SS_input, N, T, p)

            b_u_SS = my_SS[:, 0, 0].astype('int64')
            s_sk_SS = my_SS[:, 1, 0].astype('int64')
            # print np.shape(my_SS)

            ## checking BGW encoding & decoding
            # print '[ rank= ',rank,'] b_u=', b_u
            # print '[ rank= ',rank,'] b_u_SS', b_u_SS
            # temp = np.reshape(b_u_SS,(N,1))
            # print BGW_decoding(temp[0:T+1,:],range(T+1),p)

            # 1.3. Send the SS to the server
            comm.Send(b_u_SS, dest=0)
            comm.Send(s_sk_SS, dest=0)

            # 1.4. Receive the other users' SS from the server

            b_u_SS_others = np.empty(N, dtype='int64')
            s_sk_SS_others = np.empty(N, dtype='int64')

            comm.Recv(b_u_SS_others, source=0)
            comm.Recv(s_sk_SS_others, source=0)

            # Round 2. Masked Input Collection
            comm.Barrier()
            y_u = my_model

            mask = np.zeros(d, dtype='int')

            my_idx = rank -1
            my_group_idx = get_group_idxset(N,my_idx)

            for i in my_group_idx:
                if i != -1:
                    temp = np.zeros(d, dtype='int')
                    if my_idx == i:
                        np.random.seed(b_u)
                        temp = np.random.randint(0, p, size=d).astype(int)
                    elif my_idx > i:
                        np.random.seed(s_uv[i])
                        temp = np.random.randint(0, p, size=d).astype(int)
                    elif my_idx < i:
                        np.random.seed(s_uv[i-1])
                        temp = -np.random.randint(0, p, size=d).astype(int)

                    y_u = np.mod(y_u + temp, p)

            # if rank == 1:
            #    print '############mask @ rank1', mask

            comm.Send(y_u, dest=0)

            if is_sleep:
                data_size = d * 32 # bit
                comm_time = data_size / comm_mbps / (2**20) #sec

                # print 'sleep time:',comm_time
                time.sleep(comm_time)

            # print '[ rank= ',rank,'] seed=', b_u
            # print '[ rank= ',rank,']', temp

            # Round 3. Unmasking
            comm.Barrier()
            # 3.0. Rx drop_info
            drop_info = np.empty(N, dtype='int')
            comm.Recv(drop_info, source=0)

            # 3.1. Send SS
            SS_info = np.empty(N, dtype='int64')
            for i in range(N):
                if drop_info[i] == 0:
                    SS_info[i] = b_u_SS_others[i]
                else:
                    SS_info[i] = s_sk_SS_others[i]
            comm.Send(SS_info, dest=0)
            comm.Barrier()
    if rank == 0:
        time_avg = time_avg / n_trial
        print('total running time (sec) =', np.sum(time_avg))
        print('time for round 0, 1, 2, 3 (sec) =', time_avg)

        result_set = {'N': N,
                      'd': d,
                      'time_set': time_avg,
                      't_total': np.sum(time_avg),
                      'drop_rate': drop_rate}

        time_out.append(result_set)

        print('final sum=', z, "\n")
        print('# of dropout users=', np.sum(drop_info))
        print('                     drop_rate=', drop_rate, 'end!!')
        print('#############################################################################')
        print('\n')

if rank == 0:
    pickle.dump(time_out, open('./SecAggPlus_N' + str(N) + '_d' + str(d), 'wb'), -1)
