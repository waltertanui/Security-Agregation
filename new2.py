import numpy as np
from mpi4py import MPI
import logging
import sys
import pickle
import time
import gc
from sec_agg.mpc_function import LCC_encoding_w_Random_partial, LCC_decoding

# Initialize MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Logging configuration
logging.basicConfig(level=logging.DEBUG,
                    format='%(process)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s',
                    datefmt='%Y-%m-%d,%H:%M:%S')

# Process command line arguments
if len(sys.argv) == 1:
    if rank == 0:   
        logging.info("ERROR: please input the number of workers")
    exit()
elif len(sys.argv) == 2:
    N = int(sys.argv[1])
    d = 1000000
    is_sleep = False
elif len(sys.argv) == 3:
    N = int(sys.argv[1])
    d = int(sys.argv[2])
    is_sleep = False
elif len(sys.argv) == 4:  
    N = int(sys.argv[1])
    d = int(sys.argv[2])
    is_sleep = True
    comm_mbps = float(sys.argv[3])
else:
    if rank ==0: 
        logging.info("ERROR: please check the input arguments")
    exit()

# System parameters
T = int(np.floor(N/2))
U_array = np.array([T + 1]).astype('int')
p = 2**31-1
n_trail = 3
K = int(0.01*d)
M = 40
drop_rate = 0.1
time_out = []

# Calculate chunk size to reduce memory usage
chunk_size = d // M  # Size of each shard
d = chunk_size * M  # Ensure d is divisible by M

if size != N+1:
    logging.info("Number Error! N Users and 1 Server.")

if rank == 0:
    logging.info(U_array)

for t in range(len(U_array)):
    U = np.max([U_array[t], T+1])
    U1 = N
    surviving_set1 = np.array(range(U1))
    surviving_set2 = np.array(range(U))
    time_avg = np.zeros((5), dtype=float)
    
    if rank == 0:
        logging.info(f"N={N}, U={U}, T={T} start!!")

    for avg_idx in range(n_trail):
        comm.Barrier()
        logging.info('START!')

        # Server
        if rank == 0:
            comm.Barrier()
            t_total_start = time.time()
            logging.info('Server START!')
            
            # Generate alpha and beta values
            alpha = np.random.randint(1, p, size=N, dtype=np.int64)
            beta = np.random.randint(1, p, size=M+T, dtype=np.int64)
            logging.info("Server: alpha and beta are ready!")
            
            t0_comm = time.time()
            # Broadcast alpha and beta to all users
            for user_idx in range(1, N+1):
                try:
                    comm.Send([np.ascontiguousarray(alpha, dtype=np.int64), MPI.INT64_T], dest=user_idx)
                    comm.Send([np.ascontiguousarray(beta, dtype=np.int64), MPI.INT64_T], dest=user_idx)
                    logging.info(f'[rank={user_idx}] is sending alpha and beta')
                except Exception as e:
                    logging.error(f"Error sending to rank {user_idx}: {str(e)}")
                    raise
            logging.info("Server finished sending alpha and beta")
            
            # Online phase round1: Receive masked gradients
            comm.Barrier()
            masked_gradients = np.zeros((U1, K), dtype=np.int64)
            rx_req = [None]*U1
            for i in surviving_set1:
                rx_rank = i+1
                logging.info(f"rx_rank={rx_rank}")
                rx_req[i] = comm.Irecv([masked_gradients[i,:], MPI.INT64_T], source=rx_rank)
            try:
                MPI.Request.Waitall(rx_req)
            except Exception as e:
                logging.error(f"Error in Waitall_1: {str(e)}")
                raise
            
            
            masked_gradients_sum = np.mod(np.sum(masked_gradients, axis=0), p)
            
            # Online phase round2: Receive phi_alpha_j and decode
            phi_alpha_j_buffer = np.zeros((U, K), dtype=np.int64)
            rx_req = [None]*U
            for i in surviving_set2:
                rx_rank = i+1
                rx_req[i] = comm.Irecv([phi_alpha_j_buffer[i,:], MPI.INT64_T], source=rx_rank)
            try:
                MPI.Request.Waitall(rx_req)
            except Exception as e:
                logging.error(f"Error in Waitall_2: {str(e)}")
                raise

            t_comm = time.time() - t0_comm
            t0_dec = time.time()
            
            # Decoding using Lagrange interpolation
            alpha_s_eval = alpha[surviving_set2]
            beta_s = beta[:M]
            
            # Compute Lagrange coefficients for decoding
            U_dec = np.zeros((M, U), dtype=np.int64)
            for m in range(M):
                for j in range(U):
                    numerator = np.int64(1)
                    denominator = np.int64(1)
                    for k in range(U):
                        diff1 = (beta[m] - alpha_s_eval[k]) % p
                        diff2 = (alpha_s_eval[j] - alpha_s_eval[k]) % p
                        numerator = (numerator * diff1) % p
                        denominator = (denominator * diff2) % p
                    
                    try:
                        inv_denominator = pow(int(denominator), int(p-2), int(p))
                        U_dec[m,j] = (int(numerator) * inv_denominator) % p
                    except Exception as e:
                        logging.error(f"Modular inverse error: {str(e)}")
                        raise
            
            # Perform the decoding
            f_recon = np.mod(U_dec.dot(phi_alpha_j_buffer.T), p)
            
            t_dec = time.time() - t0_dec
            t_total = time.time() - t_total_start
            
            # Collect timing information from users
            time_users = np.zeros(2, dtype='float')
            for i in range(N):
                rx_rank = i + 1
                tmp = np.empty(2, dtype='float')
                #comm.Recv([tmp,MPI.FLOAT], source=rx_rank)
                #time_users += tmp
                try:
                    comm.Recv([tmp, MPI.FLOAT], source=rx_rank)
                    time_users += tmp
                except Exception as e:
                    logging.error(f"Error receiving from rank {rx_rank}: {str(e)}")
                    raise
                
            time_users = time_users / N

            time_set = np.array([t_total, time_users[0], time_users[1], t_comm, t_dec])
            logging.info('%d-th trial time info=%s' % (avg_idx, time_set))
            time_avg += time_set

            if avg_idx == n_trail-1:
                comm.Barrier()

        # User code
        elif rank <= N:
            comm.Barrier()
            t_offline_start = time.time()
            
            # Receive alpha and beta from server
            alpha = np.empty(N, dtype=np.int64)
            beta = np.empty(M+T, dtype=np.int64)
            comm.Recv([alpha,MPI.INT64_T], source=0)
            comm.Recv([beta,MPI.INT64_T], source=0)
            logging.info(f"round_idx={avg_idx} received alpha and beta")
            
            # Generate random binary mask for rand-K sparsification
            random_indices = np.random.choice(d, K, replace=False)   #获得K个不同整数的数组，bt_i置1的索引
            # False——无放回抽样（每个元素只能被选一次）
            
            bt_i = np.zeros(d, dtype=np.int64)
            bt_i[random_indices] = 1
            #kt_i=np.sorted(random_indices)  #用户i选择的K个坐标的有序集合
            logging.info(f"round_idx={avg_idx} produced bt_i")
            
            # Generate random masks
            rt_ik = np.random.randint(0, p, size=K, dtype=np.int64)
            
            # Generate phi and psi coefficients without creating full d-dimensional arrays
            # 当指定k时，比如k=1，用户i生成的ϕ(phi)和ψ(psi)的大小取决于其a_K_it(k),n的大小(d/M)，故生成的ϕi1(alpha)的大小为d/M * 1；
            # 又因为此处的k的个数为K个，所以生成的ϕik(alpha)的大小为d/M * K;
            # 其生成的两个拉格朗日插值多项式ϕik(alpha)和ψik(beta)的项数大小取决于乘积基多项式的项数大小，alpha值未知，beta值已知，故项数大小为（M+T-1）
            # 此后，在线阶段广播掩码梯度参数指的是——用户i发送自己的掩码梯度参数x_jk_t_2给其他所有用户，而非直接发送给服务器（以实现梯度参数及其坐标隐藏）
            # 各个用户接收到其他所有用户广播的掩码梯度参数（22）后，进行（23）计算，得到编码梯度的本地聚合phi(alpha_i)并发送给Server（此时由于每个用户的alpha_i不同，因此其phi_jk(alpha_i)和psi_jk(alpha_i)不同）
            # Server接收到所有用户的不同的phi(alpha_i)后，通过拉格朗日插值重建多项式phi(alpha),并代入beta(1)——beta(M)，进行（24）计算，恢复出局部梯度之和x_agg_t
            # 要保证恢复出的局部梯度之和x_agg_t与原始梯度之和相同，需保证所使用的全部K个随机掩码rt_ik之和取模p后为0(由于p过大，部分随机掩码为负数，其和即为0)
            phi_coeff = np.zeros((K, N), dtype=np.int64)
            psi_coeff = np.zeros((K, N), dtype=np.int64)
            
            # For each selected coordinate, compute its contribution to each user's share
            for k in range(K):
                coord = random_indices[k]
                # Find which shard this coordinate belongs to
                shard_idx = coord // chunk_size
                local_idx = coord % chunk_size
                
                # Compute Lagrange basis for this coordinate
                basis = np.zeros(M+T, dtype=np.int64)
                for n in range(M+T):
                    numerator = np.int64(1)
                    denominator = np.int64(1)
                    for m in range(M+T):
                        if m != n:

                            # Convert to Python int before pow()
                            alpha_py = int(alpha[rank-1])
                            beta_m_py = int(beta[m])
                            beta_n_py = int(beta[n])

                            #numerator = (numerator * (alpha_py - beta_m_py)) % p
                            #denominator = (denominator * (beta_n_py - beta_m_py)) % p
                    #denominator_py = int(denominator)
                            diff1 = (alpha_py - beta_m_py) % p
                            diff2 = (beta_n_py - beta_m_py) % p
                            numerator = (numerator * diff1) % p
                            denominator = (denominator * diff2) % p
                    #inv_denominator = pow(denominator_py, p-2, p)
                    #basis[n] = (int(numerator) * inv_denominator) % p
                    try:
                            inv_denominator = pow(int(denominator), int(p-2), int(p))
                            basis[n] = (int(numerator) * inv_denominator) % p
                    except Exception as e:
                            logging.error(f"Modular inverse error: {str(e)}")
                            raise
                
                # For real shards (first M)
                if shard_idx < M:
                    phi_coeff[k, :] = basis[shard_idx]
                
                # For random shards (last T)
                # We'll handle these when we generate the random vectors
                
                # Generate psi coefficients
                psi_coeff[k, :] = (phi_coeff[k, :] * int(rt_ik[k])) % p
            
            # Generate random vectors for redundancy (without storing full arrays)
            # We only need to compute their contribution to phi_coeff
            for t_idx in range(T):
                # Random contribution to each user's share
                random_contribution = np.random.randint(0, p, size=N, dtype=np.int64)
                for k in range(K):
                    phi_coeff[k, :] = (phi_coeff[k, :] + random_contribution * basis[M + t_idx]) % p
            
            t_offline_enc = time.time() - t_offline_start
            
            # Exchange phi and psi coefficients with other users
            tx_req1 = [None]*(N-1)
            tx_req2 = [None]*(N-1)
            tx_dest = np.delete(range(N), rank-1)
            
            for j in range(len(tx_dest)):
                bf_addr = tx_dest[j]
                tx_rank = tx_dest[j]+1
                #tx_data1 = phi_coeff[bf_addr,:].astype(int)
                #tx_data2 = psi_coeff[bf_addr,:].astype(int)
                tx_data1 = np.ascontiguousarray(phi_coeff[bf_addr,:], dtype=np.int64)
                tx_data2 = np.ascontiguousarray(psi_coeff[bf_addr,:], dtype=np.int64)
                logging.info(f'[rank={rank}] Send phi_coeff and psi_coeff to rank={tx_rank}')
                tx_req1[j] = comm.Isend([tx_data1, MPI.INT64_T], dest=tx_rank)
                tx_req2[j] = comm.Isend([tx_data2, MPI.INT64_T], dest=tx_rank)
                    
            if is_sleep:
                comm_time_per_tx = 2*N/ comm_mbps / (2**20) * 32
                comm_time = (N-1)* comm_time_per_tx
                time.sleep(comm_time)
            
            # Receive phi and psi coefficients from other users
            rx_req = [None]*(N-1)
            rx_req2 = [None]*(N-1)
            rx_source = np.delete(range(N), rank-1)
            for i in range(len(rx_source)):
                bf_addr = rx_source[i]
                rx_rank = rx_source[i]+1
                rx_req[i] = comm.Irecv([phi_coeff[bf_addr,:], MPI.INT64_T], source=rx_rank)
                rx_req2[i] = comm.Irecv([psi_coeff[bf_addr,:], MPI.INT64_T], source=rx_rank)

            logging.info(f'round_idx={avg_idx} [rank={rank}] is Waiting for phi_coeff and psi_coeff')
            try:
                MPI.Request.Waitall(tx_req1)
                MPI.Request.Waitall(tx_req2)
                MPI.Request.Waitall(rx_req)
                MPI.Request.Waitall(rx_req2)

            except Exception as e:
                logging.error(f"Error in MPI.Request.Waitall_3: {str(e)}")
                raise
            t_offline_total = time.time()-t_offline_start
             
            # Online phase round1
            time.sleep(5)  # Simulate training time

            comm.Barrier()
            t0_round1 = time.time()
            
            # Generate local gradient and apply sparsification
            local_gradient = np.random.randn(d).astype(np.float64)
            sparse_gradient = (bt_i * local_gradient).astype(np.int64)
            sparse_gradient_in_fp = sparse_gradient % p
            
            # Create masked gradient
            masked_gradient = np.zeros(K, dtype=np.int64)
            for k in range(K):
                masked_gradient[k] = (sparse_gradient_in_fp[random_indices[k]] - rt_ik[k]) % p

            # Send masked gradient to server
            my_idx = rank-1
            if my_idx in surviving_set1:
                #masked_gradient = np.mod(masked_gradient, p).astype(int)
                #logging.info(f"({my_idx}) send masked_gradient to Server")
                #comm.Send([masked_gradient, MPI.INT], dest=0)
                send_buffer = np.ascontiguousarray(masked_gradient, dtype=np.int64)
                try:
                    comm.Send([send_buffer, MPI.INT64_T], dest=0)
                    logging.info(f"({my_idx}) send masked_gradient to Server")
                except Exception as e:
                    logging.error(f"Error sending masked gradient: {str(e)}")
                    raise

            if is_sleep:
                comm_time = K/ comm_mbps / (2**20) * 32
                time.sleep(comm_time)

            t_round1 = time.time() - t0_round1

            # Online phase round2: Compute and send phi_alpha_i
            phi_alpha_i = np.zeros(K, dtype=np.int64)
            for k in range(K):
                temp=0
                for j in range(N):
                    #term1 = np.int64(phi_coeff[k,j]) * np.int64(masked_gradient[k])
                    #term1 = term1 % p  # Reduce modulo p after each operation
                    #term2 = np.int64(psi_coeff[k,j])
                    #temp += (term1 + term2) % p
                    #temp = temp % p  # Keep the sum within bounds
                    term1 = (np.int64(phi_coeff[k,j]) % p)
                    term2 = (np.int64(masked_gradient[k]) % p)
                    product = (term1 * term2) % p
                    term3 = (np.int64(psi_coeff[k,j]) % p)
                    temp = (temp + product + term3) % p
                phi_alpha_i[k] = temp


                #phi_alpha_i[k] = temp % p

            # Send phi_alpha_i to server
            if my_idx in surviving_set2:
                #logging.info(f"({my_idx}) send phi_alpha_i to Server")
                #comm.Send([phi_alpha_i, MPI.INT], dest=0)
                send_buffer = np.ascontiguousarray(phi_alpha_i, dtype=np.int64)
                try:
                    comm.Send([send_buffer, MPI.INT64_T], dest=0)
                    logging.info(f"({my_idx}) send phi_alpha_i to Server")
                except Exception as e:
                    logging.error(f"Error sending phi_alpha_i: {str(e)}")
                    raise

            if is_sleep:
                data_size = K
                comm_time = data_size/ comm_mbps / (2**20) * 32
                time.sleep(comm_time)
     
            # Send timing information to server
            time_set = np.array([t_offline_enc, t_offline_total], dtype=np.float64)
            try:
                comm.Send([time_set,MPI.DOUBLE], dest=0)
            except Exception as e:
                logging.error(f"Error sending timing info: {str(e)}")
                raise

            if avg_idx == n_trail-1:
                comm.Barrier()

    # Process and save results
    if rank == 0:
        logging.info("experimental results:")
        time_avg = time_avg / n_trail
        logging.info(time_avg)
        logging.info('total running time (sec) = %s' % str(time_avg[0]))
        logging.info('time for offline = %s' % str(time_avg[2]))

        result_set = {'N': N,
                      'U': U,
                      'time_set': time_avg
                      }
        time_out.append(result_set)
        logging.info('N,U,T= %d, %d, %d'% (N, U, T))
        logging.info('#############################################################################')

    if rank == 0:
        pickle.dump(time_out, open('./Light_N' + str(N) + '_d' + str(d), 'wb'), -1)