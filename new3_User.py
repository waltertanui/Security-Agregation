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

# Add ordered logging function
def ordered_log(comm, message):
    """Log messages in order of rank"""
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    for i in range(size):
        if i == rank:
            logging.info(message)
        comm.Barrier()

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
    d = 1000
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
# 定义生成拉格朗日插值多项式phi_ik(alpha)和psi_ik(alpha)函数
# def generate_lagrange_polynomials(alpha, beta, a_shards, rt_ik, v_ikn, u_ikn, M, T, N, p):
    #User i 生成的phi_ik的大小为K,一共有N个用户，故：
    #phi=np.zeros((K,N),dtype=np.int64)
    #psi=np.zeros((K,N),dtype=np.int64)
    #for k in range(K):
     #   for j in range(N):
      #      phi_jk=0
       #     psi_jk=0
            #前M项
        #    for n in range(M):  
         #       phi_numerator=1
                #psi_numerator=1
          #      phi_denominator=1
                #psi_denominator=1
           #     for m in range(M+T):
            #        if m!=n: #若只考虑乘积式，phi的分子与分母与psi相同
             #           phi_numerator=(phi_numerator*(alpha[j]-beta[m])) % p
                        #psi_numerator=(psi_numerator*(alpha[j]-beta[m]))%p
              #          phi_denominator=(phi_denominator*(beta[n]-beta[m]))%p
                        #psi_denominator=(psi_denominator*(beta[n]-beta[m]))%p
                #为方便后续除法的取模p，对所有分母denominator进行逆元操作
               # phi_denominator_inv = pow(int(phi_denominator),int(p-2),int(p))
                #psi_denominator_inv = pow(int(psi_denominator),int(p-2),int(p))
                            
                #a_shards[k][n]表示a_kt_i[k]的第n个分片,大小为d//M
#                a_val=np.max(a_shards[k][n])
 #               l_phi=(phi_numerator*phi_denominator_inv)%p
                #l_psi=(phi_numerator*psi_denominator_inv)%p
  #              phi_jk=(phi_jk+a_val*l_phi)%p
   #             psi_jk=(psi_jk+a_val*rt_ik[k]*l_phi)%p
            #后T项
    #        for n in range(M,M+T):
     #           phi_numerator=1
                #psi_numerator=1
      #          phi_denominator=1
                #psi_denominator=1
       #         for m in range(M+T):
        #            if m!=n:
         #               phi_numerator=(phi_numerator*(alpha[j]-beta[m])) % p
                        #psi_numerator=(psi_numerator*(alpha[j]-beta[m]))%p
          #              phi_denominator=(phi_denominator*(beta[n]-beta[m]))%p
                        #psi_denominator=(psi_denominator*(beta[n]-beta[m]))%p
                #为方便后续除法的取模p，对所有分母denominator进行逆元操作
           #     phi_denominator_inv = pow(int(phi_denominator),int(p-2),int(p))
                #psi_denominator_inv = pow(int(psi_denominator),int(p-2),int(p))
            #    l_phi=(phi_numerator*phi_denominator_inv)%p
                #l_psi=(phi_numerator*psi_denominator_inv)%p
             #   phi_jk=(phi_jk+v_ikn[k][n-M]*l_phi)%p
              #  psi_jk=(psi_jk+u_ikn[k][n-M]*l_phi)%p
                        
 #           phi[k][j]=phi_jk
  #          psi[k][j]=psi_jk
   # return phi, psi

def generate_lagrange_polynomials(alpha, beta, a_shards, rt_ik, v_ikn, u_ikn, M, T, N, p):
    K=len(a_shards)
    phi=np.zeros((K,N),dtype=np.int64)
    psi=np.zeros((K,N),dtype=np.int64)
    phi,psi=hide_coordinates(phi,psi,K,N,p)
    # 预计算拉格朗日基的分母
    #denominator = np.ones((M+T, M+T), dtype=np.int64)
    #denominator = 1
    #for n in range(M+T):
        
     #   for j in range(M+T):
      #      if n!=j:
       #         denominator =( denominator * (beta[n] - beta[j] )) % p

    # 计算分母的模逆元
    #inv_denominator=np.array([[pow(int(denominator[n][j]),int( p-2), int(p)) if j!=n else 0 for j in range(M+T)] for n in range(M+T)])
    #inv_denominator = pow(int(denominator),int(p-2),int(p))
    # 计算拉格朗日基
    for k in range(K):
        for j in range(N):
            phi_jk=0
            psi_jk=0
            
            # 计算分子项
            #numerator=np.ones((M+T), dtype=np.int64)
            for m in range(M+T):
                l_m=1
                
                for n in range(M+T):
                    if n!=m:
                        l_m=(l_m*(alpha[j]-beta[n])) % p
                # 计算分母的模逆元
                denominator=1
                for n in range(M+T):
                    if n!=m:
                        denominator =( denominator * (beta[m] - beta[n] )) % p
                inv_denominator = pow(int(denominator),int(p-2),int(p))
                
                l_m=(l_m*inv_denominator)%p
                

                if m<M: #前M项
                    a_val = np.max(a_shards[k][m]) % p
                    phi_jk = (phi_jk + a_val * l_m) % p
                    psi_jk = (psi_jk + a_val * rt_ik[k] * l_m) % p

                else: #后T项
                    phi_jk = (phi_jk + v_ikn[k][m-M] * l_m) % p
                    psi_jk = (psi_jk + u_ikn[k][m-M] * l_m) % p

            phi[k][j]=phi_jk
            psi[k][j]=psi_jk
    return phi, psi


 #多项式插值重建
def interpolate(alpha_s_eval,phi_alpha_buffer,beta,M,p):
    U_dec=np.zeros((M,len(alpha_s_eval)),dtype=np.int64)
    for m in range(M):
        for j in range(len(alpha_s_eval)):
            numerator=1
            denominator=1
            for k in range(len(alpha_s_eval)):
                if k!=j:
                    numerator=(numerator*(beta[m]-alpha_s_eval[k])) % p
                    denominator=(denominator*(alpha_s_eval[j]-alpha_s_eval[k])) % p
            inv_denominator = pow(int(denominator),int(p-2),int(p))
            
            U_dec[m][j]=(numerator*inv_denominator)%p
    return U_dec.dot(phi_alpha_buffer.T)%p

# 服务器聚合逻辑
def server_aggregate(phi_alphas,alpha_s_eval,beta,M,p):
    #phi_alphas——从用户端接收的phi(alpha_i)的值
    #alpha_s_eval——幸存用户的alpha值
    #beta——公共参数
    U=len(alpha_s_eval)
    if U<M+T:
        #raise ValueError("NO enough surviving users for reconstruction!")
        logging.error(f"Not enough surviving users (got {U}, need at least {M+T})")
        return None
        
    # 只使用前M+T个用户的输入
    if U > M + T:
        phi_alphas = phi_alphas[:M+T]
        alpha_s_eval = alpha_s_eval[:M+T]

    # 构建插值矩阵
    A=np.zeros((M,U),dtype=np.int64)
    for m in range(M):
        for j in range(U):
            numerator=1
            denominator=1
            for k in range(U):
                if k!=j:
                    numerator=(numerator*(beta[m]-alpha_s_eval[k])) % p
                    denominator=(denominator*(alpha_s_eval[j]-alpha_s_eval[k])) % p
            inv_denominator = pow(int(denominator),int(p-2),int(p))
            A[m][j]=(numerator*inv_denominator)%p
    # 计算聚合梯度
    x_agg = (A @ phi_alphas) % p
    return x_agg

#添加额外噪声对坐标进行隐藏
def hide_coordinates(phi,psi,K,N,p):
    random_noise_phi=np.random.randint(0,p,size=(K,N),dtype=np.int64)
    random_noise_psi=np.random.randint(0,p,size=(K,N),dtype=np.int64)
    
    # 确保噪声在聚合时能够抵消
    for k in range(K):
        noise_sum_phi=np.sum(random_noise_phi[k,:])%p
        random_noise_phi[k,0] = (random_noise_phi[k,0] - noise_sum_phi) % p
        noise_sum_psi=np.sum(random_noise_psi[k,:])%p
        random_noise_psi[k,0] = (random_noise_psi[k,0] - noise_sum_psi) % p

    phi=(phi+random_noise_phi)%p
    psi=(psi+random_noise_psi)%p
    return phi,psi

# 添加有限域转换函数
def real_to_finite_field(x,scale=1e6,p=2**31-1):
    # round——四舍五入
    #首先缩放并四舍五入
    scaled=np.round(x*scale).astype(np.int64)
    # 然后取模以限制在有限域内
    return np.mod(scaled,p)

# 添加验证步骤
def verify_aggregation(x_agg,phi_alphas,surviving_set,p):
    if np.any(x_agg<0) or np.any(x_agg>=p):
        logging.error("Aggregation result is out of bounds!")
        return False
    
    return True
# Server端
# System parameters
T = int(np.floor(N/2))
U_array = np.array([T + 1]).astype('int')
p = 2**31-1
n_trail = 3
K = int(0.01*d)
M = 6
drop_rate = 0.1
time_out = []

# Calculate chunk size to reduce memory usage
chunk_size = d // M  # 向下取整（d的数量很大时，可忽略误差）
d = chunk_size * M  # 保证d能够被M整除

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
        #logging.info('START!')

        # Server
        if rank == 0:
            comm.Barrier()
            t_total_start = time.time()
            logging.info('Server START!')
            logging.info(f"System parameters :round={avg_idx}, N={N}, U={U}, T={T}, d={d}, K={K}, M={M}, drop_rate={drop_rate}, chunk_size={chunk_size}")
            
            # Server生成公共参数alpha和beta
            alpha = np.random.randint(1, p, size=N, dtype=np.int64)
            beta = np.random.randint(1, p, size=M+T, dtype=np.int64)
            logging.info("Server: alpha and beta are ready!")
            
            t0_comm = time.time()
            # 将alpha和beta发送给所有用户
            #for user_idx in range(1, N+1):
             #   try:
              #      comm.Send([np.ascontiguousarray(alpha, dtype=np.int64), MPI.INT64_T], dest=user_idx)
               #     comm.Send([np.ascontiguousarray(beta, dtype=np.int64), MPI.INT64_T], dest=user_idx)
                #    logging.info(f'Server is sending alpha and beta to [rank={user_idx}]')
                #except Exception as e:
                 #   logging.error(f"Error sending to rank {user_idx}: {str(e)}")
                  #  raise
            alpha_reqs=[]
            beta_reqs=[]
            for user_idx in range(1,N+1):
                # 发送alpha
                req = comm.Isend([np.ascontiguousarray(alpha, dtype=np.int64), MPI.INT64_T], dest=user_idx)
                alpha_reqs.append(req)
                # 发送beta
                req = comm.Isend([np.ascontiguousarray(beta, dtype=np.int64), MPI.INT64_T], dest=user_idx)
                beta_reqs.append(req)
                logging.info(f'Server is sending alpha and beta to [rank={user_idx}]')
            MPI.Request.Waitall(alpha_reqs)
            MPI.Request.Waitall(beta_reqs)
            logging.info("Server finished sending alpha and beta")
            #comm.Barrier() #同步，确保所有User开始接收
            # 服务器端在线阶段第二轮
            # 接收所有幸存用户发送的编码梯度的本地聚合phi(alpha_i)——至少需要M+T个用户的值
            
            #接收所有幸存用户发送的phi_alpha_i，创建一个缓存器phi_alpha_buffer,大小为U*K
            # Server starts to receive phi_alpha!
            logging.info("Server starts to receive phi_alpha!")
            phi_alpha_buffer=np.zeros((U,K)).astype(np.int64)
            
            # Create non-blocking receive requests for all surviving users
            rx_reqs = []
            for j in surviving_set2:  # surviving_set2 contains 0-based indices
                rx_req = comm.Irecv([phi_alpha_buffer[j,:], MPI.INT64_T], source=j+1)
                rx_reqs.append(rx_req)
                logging.info(f"Server created receive request for user {j+1}")
            
            # Wait for all receives to complete
            MPI.Request.Waitall(rx_reqs)
            logging.info("Server received all phi_alpha values")
            
            t_comm = time.time() - t0_comm
            #多项式插值重建，得到phi(alpha)
            t0_dec = time.time()
            alpha_s_eval=alpha[surviving_set2]
            beta_s=beta[:M]
            U_dec=np.zeros((M,U),dtype=np.int64)
#            for m in range(M):
 #               for j in range(U):
  #                  numerator=1
   #                 denominator=1
    #                for k in range(U):
     #                   if k!=j:
      #                      numerator=numerator*(beta[m]-alpha_s_eval[k])%p
       #                     denominator=denominator*(alpha_s_eval[j]-alpha_s_eval[k])% p
        #            inv_denominator=pow(int(denominator),int(p-2),int(p))
         #           U_dec[m,j]=(numerator*inv_denominator) % p
            #代入beta(1)——beta(M)，按公式（24）计算出x_agg_t

            #x_agg_t=interpolate(alpha_s_eval,phi_alpha_buffer,beta_s,M,p)
            x_agg=server_aggregate(phi_alpha_buffer,alpha_s_eval,beta_s,M,p)

            #x_agg_t=np.mod(U_dec.dot(phi_alpha_buffer.T),p)
            logging.info(f"Aggregated gradient: {x_agg}")

            # 验证聚合结果是否在有限域内
            if not verify_aggregation(x_agg, phi_alpha_buffer, surviving_set2, p):
                logging.error("Aggregation verification failed!")
                

            #----------------------------------------------------------------------
            #----------------------------------------------------------------------

            # Perform the decoding
            
            t_dec = time.time() - t0_dec
            t_total = time.time() - t_total_start
            #x_agg_t=np.mod(U_dec.dot(phi_alpha_buffer.T),p)
            #logging.info(f"Aggregated gradient: {x_agg_t}")
            #----------------------------------------------------------------------

            # Perform the decoding
            
            t_dec = time.time() - t0_dec
            t_total = time.time() - t_total_start
            
            # Collect timing information from users
            time_users = np.zeros(2, dtype='float')
            for i in range(N):
                rx_rank=i+1
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

        # 用户端
        elif rank <= N:
            if rank != 0:
                comm.Barrier() # Barrier 1: Only user processes sync here
            t_offline_start = time.time()
            # 生成二进制向量a_k，大小为d，仅第k个元素为1，其余为0
            # 并分片——此处后移至kt_i之后
            #for k in range(d):
             #   a_k=[0]*d
                #a_k[k]=1
               # a_shards=np.zeros((d,M),dtype=np.int64)
                #a_shards[k] = np.split_into_shards(a_k, M)
                # 此时a_shards[1]为a_k的第二个分片，a_shards[1][k]为a_k的第二个分片中的第k个元素

            # 用户同意N+M+T个不同的公共参数（接收Server发送的公共参数alpha和beta）,值为0-p-1
            alpha = np.ascontiguousarray(np.empty(N, dtype=np.int64))
            beta = np.ascontiguousarray(np.empty(M+T, dtype=np.int64))
            #np.ascontiguousarray——创建可写缓冲区
            
            #comm.Recv([alpha,MPI.INT64_T], source=0)
            #comm.Recv([beta,MPI.INT64_T], source=0)
            #logging.info(f" User {rank} received alpha and beta")
            #comm.Barrier()
           
            #修改为非阻塞式
            reqs=[]
            alpha_req=comm.Irecv([alpha,N,MPI.INT64_T], source=0)
            beta_req=comm.Irecv([beta,M+T,MPI.INT64_T], source=0)
            reqs.extend([alpha_req,beta_req])
            MPI.Request.Waitall(reqs) # Users wait here until alpha/beta received from server

            # Add a log here to confirm reception
            logging.info(f"User {rank} finished receiving alpha and beta.")

            if rank != 0:
                comm.Barrier() # Barrier 2: Only user processes sync here
            #for i in range(1,N+1): #按照rank顺序输出日志
            #   if i==rank:
            #      logging.info(f"User {rank} received alpha and beta")
            if rank!=0:
                # Replace ordered_log with standard logging to avoid deadlock
                # ordered_log(comm, f"User {rank} received alpha and beta")
                logging.info(f"User {rank} passed alpha/beta reception barrier.")

            #设置独立随机种子
            np.random.seed(rank*1000+int(time.time()*1000)%1000000)
            # 生成随机二进制掩码bt_i,大小为d，仅K个元素为1，其余为0 —— 用于rand-K稀疏化
            random_indices = np.random.choice(d, K, replace=False) # 从0-d-1中随机选择K个不同的整数，大小为K，False——无放回抽样（每个元素只能被选一次）
            bt_i = np.zeros(d, dtype=np.int64)
            bt_i[random_indices] = 1

            # 获得K个坐标的有序集合
            kt_i=np.sort(random_indices)

            for i in range(1,N+1):
                if i==rank:
                    logging.info(f"round_idx={avg_idx}, User {rank} : random_indices = {random_indices}, kt_i = {kt_i}")
            logging.info("kt_i all be logged")
            #comm.Barrier() 
            #logging.info(f"round_idx={avg_idx}, rank={rank} : selected coordinates after sorting = {kt_i}")
            
            # 按公式（19）、（20）生成两个拉格朗日插值多项式phi_ik(alpha)和psi_ik(alpha)

            # 首先，计算拉格朗日基多项式（乘积的式子）

            # 然后，获得a_kt_i[k]_n的值——将a_kt_i[k]分成M个分片后，选择第n个分片，即a_shards[n-1]，其大小为d/M
            # kt_i[k]表示random_indices中第k个升序元素的值，即：选择的a_k中的k为kt_i[k]
            # _n表示将a_kt_i[k]分成M个分片后，选择第n个分片，即a_shards[n-1]
            
            # 生成二进制向量a_k，大小为d，仅第k个元素为1，其余为0
            # 并分片
            #chunk_size = d // M
            a_shards=np.zeros((K,M,chunk_size),dtype=np.int64)
            for k in range(K):
                a_k=np.zeros(d, dtype=np.int64)
                a_k[kt_i[k]]=1
                #a_shards的大小为(K,M,d//M),即为：K*M的矩阵，每个元素为一个d//M大小的矩阵
                # 如：kt_i={2,6},则a_shards[0]=((0,0,1),(0,0,0),(0,0,0));a_shards[1]=((0,0,0),(0,0,0),(1,0,0))

                # 将a_k分成M个分片
                for m in range(M):
                    start_idx=m*chunk_size
                    end_idx=(m+1)*chunk_size if m!=M-1 else d
                    a_shards[k,m]=a_k[start_idx:end_idx]
                    #logging.info(f"round_idx={avg_idx}, rank={rank} : a_shards[{k},{m}] = {a_shards[k,m]}")
                #logging.info(f"round_idx={avg_idx}, rank={rank} : a_shards[{k}] = {a_shards[k]}")

            #每个用户使用不同的种子
            local_seed=rank*1000+int(time.time()*1000)%1000000
            rng=np.random.RandomState(local_seed)
            rt_ik=rng.randint(0,p,size=K,dtype=np.int64)
            logging.info(f"round_idx={avg_idx} , rank={rank} generated initial rt_ik") # Log initial generation

            # Synchronize before rt_ik exchange to ensure all users generated initial rt_ik
            comm.Barrier() 
            logging.info(f"User {rank} passed first rt_ik barrier")

            #确保所有用户掩码之和为0
            if rank==N:
                total=np.zeros(K, dtype=np.int64)
                received_rts = {} # Store received values for logging/verification
                logging.info(f"User {rank} starting to receive rt_ik from others")
                for i in range(1,N): # Expect N-1 messages
                    rt=np.empty(K, dtype=np.int64)
                    status = MPI.Status() # Get status to find source
                    comm.Recv([rt,MPI.INT64_T], source=MPI.ANY_SOURCE, status=status) # Receive from any source
                    source_rank = status.Get_source()
                    received_rts[source_rank] = rt.copy() # Store for verification
                    logging.info(f"User {rank} received rt from User {source_rank}")
                    total=(total+rt)%p
                
                logging.info(f"User {rank} received all rts: {received_rts}") # Log all received values

                rt_ik=(-total)%p # Calculate final rt_ik for user N
                logging.info(f"User {rank} calculated final rt_ik: {rt_ik}")
                
                # Verification step on User N
                mask_sum = np.sum(rt_ik)
                for r in received_rts.values():
                    mask_sum = (mask_sum + np.sum(r)) % p
                
                if mask_sum != 0:
                    logging.error(f"Mask sum verification FAILED on User {rank}! Sum={mask_sum}")
                else:
                    logging.info(f"Mask sum verified by User {rank}: sum=0")

            elif rank!=0: # All other users (1 to N-1) send to N
                comm.Send([rt_ik,MPI.INT64_T], dest=N)
                logging.info(f"User {rank} sent rt_ik to User {N}")
            
            # Add another barrier to ensure rt_ik calculation/distribution is complete for ALL users
            comm.Barrier() 
            logging.info(f"User {rank} finished rt_ik synchronization. Final rt_ik: {rt_ik}")
            
            # 最后，生成随机噪声v_ikn和u_ikn，用于掩藏掩码值rt_ik及所选坐标
            # 继续，生成随机噪声v_ikn和u_ikn，用于掩藏掩码值rt_ik及所选坐标
            v_ikn=np.random.randint(0,p,size=(K,T),dtype=np.int64)

            u_ikn=np.random.randint(0,p,size=(K,T),dtype=np.int64)
            for i in range(1,N+1):
                if i==rank:
                    logging.info(f"round_idx={avg_idx} , rank={rank} : v_ikn = {v_ikn}, u_ikn = {u_ikn}")
            #comm.Barrier()

            # 用户端离线阶段结束
            #numerator——分子,denominator——分母
            
            phi, psi = generate_lagrange_polynomials(alpha, beta, a_shards, rt_ik, v_ikn, u_ikn, M, T, N, p)
            logging.info(f"round_idx={avg_idx} , rank={rank} : phi = {phi}, psi = {psi}")
            
            #comm.Barrier()
            #phi大小为K*N,第j列为第j个用户的phi_ik(alpha)的系数——全局缓存区
            # 此外，将每个用户生成的phi_ik(alpha)和psi_ik(alpha)（大小为K）的系数存储到全局缓存区phi和psi中，大小为K*N
            tx_req_phi=[]
            tx_req_psi=[]
            for j in range(1,N+1):
                if j !=rank:
                    req_phi=comm.Isend([phi[:,j-1].copy(), MPI.INT64_T], dest=j, tag=1)
                    req_psi=comm.Isend([psi[:,j-1].copy(), MPI.INT64_T], dest=j, tag=2)
                    tx_req_phi.append(req_phi)
                    tx_req_psi.append(req_psi)
                    logging.info(f"User %d send phi and psi to User %d"%(rank,j))
 #           tx_req_phi=[]
  #          tx_req_psi=[]
            #dest_users=[x for x in range(1,N+1) if x != rank]
   #         tx_dest=np.delete(range(N), rank - 1) # 排除自己
    #        for j in range(len(tx_dest)):  #遍历N-1次，为每个目标用户进程创建发送请求
     #           bf_addr = tx_dest[j]  #目标用户的索引(从0开始)
      #          tx_rank = tx_dest[j] + 1  #目标用户的进程编号（从1开始；0为Server）
       #         tx_data=np.ascontiguousarray(phi[:,bf_addr].astype(int))
        #        tx_data_1=np.ascontiguousarray(psi[:,bf_addr].astype(int))
         #       logging.info('[ rank= %d ] send phi and psi to %d' % (rank, tx_rank))  #记录日志，表示当前进程rank正在向目标进程tx_rank发送数据
          #      req_phi = comm.Isend([tx_data, MPI.INT64_T], dest=tx_rank,tag=1)  #向目标进程tx_rank发送数据phi
           #     req_psi = comm.Isend([tx_data_1, MPI.INT64_T], dest=tx_rank,tag=2)  #向目标进程tx_rank发送数据psi
            #    tx_req_phi.append(req_phi)
             #   tx_req_psi.append(req_psi)
                

            #for j in dest_users:
             #   req=comm.Isend([np.ascontiguousarray(phi[:,j-1]), MPI.INT64_T], dest=j, tag=1)
              #  phi_reqs.append(req)
               # req=comm.Isend([np.ascontiguousarray(psi[:,j-1]), MPI.INT64_T], dest=j, tag=2)
                #psi_reqs.append(req)
                #logging.info(f"User %d send phi and psi to User %d"%(rank,j))
            #MPI.Request.Waitall(phi_reqs)
            #MPI.Request.Waitall(psi_reqs)

            if is_sleep:
                comm_time_per_tx=K/comm_mbps / (2**20) * 32
                comm_time=comm_time_per_tx * 2 * (N-1)
                time.sleep(comm_time)
            #comm.Barrier() #确保所有发送完成
            logging.info("All phi and psi have been sent")
            #从其他用户接收phi_ik(alpha)和psi_ik(alpha)
            phi_coeff=np.zeros((K,N),dtype=np.int64)
            psi_coeff=np.zeros((K,N),dtype=np.int64)
            #直接填充自己的数据
            phi_coeff[:,rank-1]=phi[:,rank-1]
            psi_coeff[:,rank-1]=psi[:,rank-1]
            rx_req_phi=[]
            rx_req_psi=[]
            rx_buffer=[]
            for j in range(1,N+1):
                if j !=rank:
                    buf_phi = np.empty(K, dtype=np.int64)
                    buf_psi = np.empty(K, dtype=np.int64)
                    req_phi=comm.Irecv([buf_phi, MPI.INT64_T], source=j, tag=1)
                    req_psi=comm.Irecv([buf_psi, MPI.INT64_T], source=j, tag=2)
                    # 这里的tag=1和tag=2是标识符，用于区分不同类型的消息，分别代码phi和psi
                    rx_req_phi.append(req_phi)
                    rx_req_psi.append(req_psi)
                    rx_buffer.append((j-1,buf_phi,buf_psi))
                    logging.info(f"User %d received phi and psi from User %d"%(rank,j))
            MPI.Request.Waitall(tx_req_phi)
            MPI.Request.Waitall(tx_req_psi)

            MPI.Request.Waitall(rx_req_phi)
            MPI.Request.Waitall(rx_req_psi)
            #comm.Barrier()
            logging.info("All phi and psi have been received")
            for bf_addr, buf_phi, buf_psi in rx_buffer:
                phi_coeff[:,bf_addr] = buf_phi
                psi_coeff[:,bf_addr] = buf_psi
            logging.info("round_idx = %d [ rank= %d ] data phi and psi is ready..."% (avg_idx, rank))
            #comm.Barrier() # 确保所有用户接收完成
            logging.info("All data of phi and psi is ready!")
            
            # 采用非阻塞方式发送和接收数据流时，在Isend/Irecv 完成之前，不能修改发送缓存区或者释放接收缓冲区，否则可能导致数据损坏或程序崩溃


            # 最后，每个用户i代入最初获得的公共参数alpha_j，得到phi_ik(alpha_j)和psi_ik(alpha_j)，大小为K，并发送给用户j —— 在全局缓存区上表现为:
            # 遍历每个用户i∈[N]，对于全局缓存区phi_coeff和psi_coeff上的每一行j!=i，将其值更新为phi_ik(alpha_j)和psi_ik(alpha_j)
            # 此步已由前面函数中的alpha[j]实现

            # 用户端离线阶段结束
            t_offline_total=time.time()-t_offline_start
            #用户端在线阶段第一轮开始
            logging.info("User offline end!")
            #模拟局部训练时间
            time.sleep(5)
            #comm.Barrier() #同步，确保所有进程都已经训练完毕，准备开始发送掩码模型
            logging.info("User %d online phase start!"%(rank))
            
            
            t0_round1 = time.time() #记录第一轮在线阶段时间开始
            start_encoding_time = time.time() #记录编码开始时间
            #模拟最开始的本地模型xt_i（大小为d）
            xt_i = np.random.randn(d).astype(np.int64)
            #本地模型稀疏化，得到稀疏化的局部梯度xt_i_1（大小为d）
            xt_i_1 = bt_i*xt_i
            # （21）将其从实数域转换为有限域，得到xt_i_mod（大小为d）
            xt_i_mod = real_to_finite_field(xt_i_1 , p)
            # （22）每个用户i对xt_i_mod(大小为d)进行掩码（减去随机掩码rt_ik（大小为K）），得到掩码梯度参数xt_i_mod_masked，并将其广播给其他所有用户j(j!=i)(考虑采用全局掩码梯度缓冲区)
            xt_i_mod_masked=np.zeros(K).astype(np.int64) #大小为K
            for k in range(K):
                xt_i_mod_masked[k] = xt_i_mod[kt_i[k]] - rt_ik[k]

            global_masked_buffer=np.zeros((N,K)).astype(np.int64) #大小为N*K
            my_idx=rank-1
            if my_idx in surviving_set1:
                send_reqs = []
                dest_users = [x for x in range(1, N+1) if x != rank]  # 排除自己
    
                for j in dest_users:
                    req = comm.Isend([xt_i_mod_masked, MPI.INT64_T], dest=j)
                    send_reqs.append(req)
    
                # 本地数据直接存储
                global_masked_buffer[my_idx,:] = xt_i_mod_masked
                MPI.Request.Waitall(send_reqs)
                logging.info("User %d send masked_model to User %d" % (my_idx,j))

            #模拟通信延迟
            if is_sleep:
                data_size = K # 参数数量
                comm_time = data_size / comm_mbps / (2**20) * 32 #秒
                time.sleep(comm_time)  #让当前进程暂停comm_time秒，模拟通信延迟

            t_round1 = time.time() - t0_round1  #计算第一轮在线阶段总耗时
            #用户端在线阶段第二轮

            # 所有用户i接收完毕其他用户发送的掩码梯度参数xt_i_mod_masked后，按公式（23）计算出一个（所有幸存用户的）编码梯度的本地聚合phi(alpha_i)，并将其发送给Server
            if my_idx in surviving_set1:
                recv_reqs = []
                source_users = [x for x in range(1, N+1) if x != rank]  # 排除自己
    
                for j in source_users:
                    temp_buf = np.empty(K, dtype=np.int64)
                    req = comm.Irecv([temp_buf, MPI.INT64_T], source=j)
                    recv_reqs.append((req, j, temp_buf))
                    
                #MPI.Request.Waitall(recv_reqs)
                logging.info("User %d received masked_model from User %d"%(rank,j))
    
                # 处理本地数据
                phi_i = phi_coeff[:,rank-1]

                logging.info("Local data have been received!")

                # 等待所有接收完成并处理
                for req, j, temp_buf in recv_reqs:
                    req.Wait()
                    global_masked_buffer[j-1,:] = temp_buf

                logging.info("All receiving done!")

            # 按公式（23）计算出一个（所有幸存用户的）编码梯度的本地聚合phi(alpha_i)，并将其发送给Server
            phi_alpha_i=np.zeros(K).astype(np.int64) #大小为K
            
            # 确保在计算phi_alpha_i之前，所有用户都已经接收到了其他用户的掩码梯度参数
            # Only surviving users calculate and send
            if my_idx in surviving_set2:
                phi_alpha_i = np.zeros(K, dtype=np.int64)
                
                # Calculate using only surviving users' data
                for k in range(K):
                    for j in surviving_set2:
                        val = global_masked_buffer[j,k]
                        phi_val = phi_coeff[k,j]
                        psi_val = psi_coeff[k,j]
                        phi_alpha_i[k] = (phi_alpha_i[k] + val * phi_val + psi_val) % p
                    # It might be better to log calculation completion *after* the loop
                    # logging.info(f"User {rank} calculated coordinate {k}") 
                
                # Add a log message indicating calculation is complete before sending
                logging.info(f"User {rank} finished calculating all phi_alpha_i values")

                # Send complete array
                comm.Send([phi_alpha_i, MPI.INT64_T], dest=0)
                # Replace ordered_log with a standard log to avoid deadlock
                # ordered_log(comm, f"User {rank} SENT phi_alpha_i") 
                logging.info(f"User {rank} SENT phi_alpha_i to Server") # Use standard logging
            
            # Adjust this log message for clarity if needed, maybe indicate rank
            logging.info(f"User {rank} finished Online Phase round 2!") 
            
            #for my_idx in surviving_set2:
             #   for k in range(K):
             #       phi_alpha_i = (phi_alpha_i+xt_i_mod_masked[k]*phi_coeff[k,my_idx]+psi_coeff[k,my_idx]) % p
              #  comm.Send(phi_alpha_i,MPI.INT64_T,dest=0)

            if is_sleep:
                data_size = 1 # 参数数量
                comm_time = len(surviving_set2)*data_size / comm_mbps / (2**20) * 32 #秒
                time.sleep(comm_time)  #让当前进程暂停comm_time秒，模拟通信延迟

            # 用户端在线阶段结束



            #----------------------------------------------------------------------
            





            # 当指定k时，比如k=1，用户i生成的ϕ(phi)和ψ(psi)的大小取决于其a_K_it(k),n的大小(d/M)，故生成的ϕi1(alpha)的大小为d/M * 1；
            # 又因为此处的k的个数为K个，所以生成的ϕik(alpha)的大小为d/M * K;
            # 其生成的两个拉格朗日插值多项式ϕik(alpha)和ψik(beta)的项数大小取决于乘积基多项式的项数大小，alpha值未知，beta值已知，故项数大小为（M+T-1）
            # 此后，在线阶段广播掩码梯度参数指的是——用户i发送自己的掩码梯度参数x_jk_t_2给其他所有用户，而非直接发送给服务器（以实现梯度参数及其坐标隐藏）
            # 各个用户接收到其他所有用户广播的掩码梯度参数（22）后，进行（23）计算，得到编码梯度的本地聚合phi(alpha_i)并发送给Server（此时由于每个用户的alpha_i不同，因此其phi_jk(alpha_i)和psi_jk(alpha_i)不同）
            # Server接收到所有用户的不同的phi(alpha_i)后，通过拉格朗日插值重建多项式phi(alpha),并代入beta(1)——beta(M)，进行（24）计算，恢复出局部梯度之和x_agg_t
            # 要保证恢复出的局部梯度之和x_agg_t与原始梯度之和相同，需保证所使用的全部K个随机掩码rt_ik之和取模p后为0(由于p过大，部分随机掩码为负数，其和即为0)

            # 公式(19)中的phi_ik(alpha)的前一部分——分片与拉格朗日插值，将坐标信息a_k分散到M个分片中，确保Server无法直接获取完整坐标（坐标隐藏）
            # 公式(19)中的phi_ik(alpha)的后一部分——引入随机噪声v_ikn，防止敌手通过插值恢复分片信息（冗余）

            # 公式(20)中的psi_ik(alpha)——隐藏梯度参数
            # 公式(21)中的psi_ik(alpha)的前一部分——将梯度参数掩码rt_ik与坐标分片绑定，确保Server无法分离二者
            # 公式(21)中的psi_ik(alpha)的后一部分——添加冗余，进一步混淆掩码信息，增强隐私性

           