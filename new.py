import numpy as np
from mpi4py import MPI
import logging
import sys
import pickle
import time
from sec_agg.mpc_function import LCC_encoding_w_Random_partial, LCC_decoding

#初始化MPI环境
comm = MPI.COMM_WORLD #创建MPI通信器
rank = comm.Get_rank() #获取当前进程的rank（0为服务器，1~N为用户）
size = comm.Get_size() #获取总进程数

#日志配置
logging.basicConfig(level=logging.DEBUG,
                    format='%(process)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s',
                    datefmt='%Y-%m-%d,%H:%M:%S')

#处理命令行参数
if len(sys.argv) == 1: #没有提供参数，报错并退出
    if rank == 0:   
        logging.info("ERROR: please input the number of workers")
    exit()
elif len(sys.argv) == 2:
    N = int(sys.argv[1]) #提供了工作节点数量N；int——直接截断取整
    d = 1000000 #模型参数维度，默认约100万
    is_sleep = False  #不启用睡眠模式
elif len(sys.argv) == 3:
    #提供了工作节点数N和模型参数维度d
    N = int(sys.argv[1])
    d = int(sys.argv[2])
    is_sleep = False
elif len(sys.argv) == 4:  
    #提供了工作节点数N、模型参数维度d和通信带宽comm_mbps（单位：Mbps）
    N = int(sys.argv[1])
    d = int(sys.argv[2])
    is_sleep = True
    comm_mbps = float(sys.argv[3]) # unit: Mbps
else: #参数错误
    if rank ==0: 
        logging.info("ERROR: please check the input arguments")
    exit()


#N=100
T=int(np.floor(N/2))
U_array = np.array([T + 1]).astype('int') 

p=2**31-1
n_trail=3
#d=1000000
K=int(0.01*d)
M=40
drop_rate=0.1
#U_array = np.array(T + 1, np.floor((1-drop_rate)*N)).astype('int') 
#U_array = np.array([T + 1, np.floor((1-drop_rate)*N)]).astype('int')
#生成包含两个元素的数组，分别为最小安全用户数下限和期望存活用户数

#is_sleep = 1 #设置是否有通信延迟(1为有，0为没有)
#comm_mbps=100 #通信带宽，单位为Mbps

my_model = np.ones(shape=(d)).astype('int')

time_out=[]

if size!=N+1:
    logging.info("Number Error! N Users and 1 Server.")

if rank==0:
    logging.info(U_array) #记录U_array的值

for t in range(len(U_array)):
    U=np.max([U_array[t],T+1])
    d=int(np.ceil(d/M))*M#使d能够被M整除，方便后续分片
    U1 = N #用户总数
    surviving_set1 = np.array(range(U1)) #第一轮存活的节点N
    surviving_set2 = np.array(range(U)) #第二轮存活的节点U
    time_avg = np.zeros((5), dtype=float)  
    if rank==0:
        logging.info("N,U,T=',N,U,T,'start!!")
#分布式计算与通信
    for avg_idx in range(n_trail):
        comm.Barrier()
        logging.info('START!')
#Server
        if rank==0:#Server
            comm.Barrier()
            t_total_start=time.time()
            logging.info('Server START!')
            #offline
            alpha=np.random.randint(1,p,size=N,dtype=np.int64)
            beta=np.random.randint(1,p,size=M+T,dtype=np.int64)
            logging.info("Server: alpha and beta are already!")

            t0_comm = time.time() # 开始通信计时
            #将公共参数alpha和beta广播给所有用户,alpha和beta为从有限域Fp中任取的一个随机值（但是不同的随机值）
            for user_idx in range(1, N+1):
                comm.Send(alpha, dest=user_idx)
                comm.Send(beta, dest=user_idx)
                logging.info('[rank=%d] is sending alpha and beta '%(user_idx))
            logging.info("Server finished send alpha and beta")

        #online phase round1
        # 在线阶段第一轮：接收所有用户的掩码梯度 mask_gradient
        # 初始化接收缓冲区（假设每个用户发送 K 个参数，每个参数为有限域整数）
            comm.Barrier() # 确保所有用户都准备好了开始发送数据，Server也准备好了开始接收数据，保证数据接收的顺序和完整性
        
            masked_gradients = np.zeros((U1, K), dtype=int)
            rx_req=[None]*U1
            for i in surviving_set1:
                rx_rank=i+1
                logging.info("rx_rank=%d" % rx_rank)
                rx_req[i]=comm.Irecv([masked_gradients[i,:],MPI.INT],source=rx_rank)
            MPI.Request.Waitall(rx_req)
            
            masked_gradients_sum=np.mod(np.sum(masked_gradients,axis=0),p)
                       
        #online phase round2

        # 在线阶段第二轮：Server接收数据并恢复聚合梯度
        
        # 接收所有用户的phi_alpha_j

            #phi_alpha_j_buffer=np.zeros((U,K).astype(np.int64),dtype=int)
            phi_alpha_j_buffer=np.zeros((U,K),dtype=int)
            #从存活用户接收phi_alpha_j
            rx_req=[None]*U

            for i in surviving_set2:
                rx_rank=i+1
                rx_req[i]=comm.Irecv([phi_alpha_j_buffer[i,:],MPI.INT],source=rx_rank)
            MPI.Request.Waitall(rx_req)

            t_comm = time.time() - t0_comm #记录通信时间结束
            t0_dec = time.time() #记录解码时间开始

        #可选方案——调用LCC_decoding()函数进行解码
            #准备评估点alpha的值（幸存用户）
            alpha_s_eval = alpha[surviving_set2]
            #准备目标点beta的值（前M个）
            beta_s = beta[:M]
            #重塑phi_alpha_j_buffer来使其匹配LCC_decoding函数的输入要求：f_eval 大小为 [RT x d]

            f_eval = phi_alpha_j_buffer.T  #将phi_alpha_j_buffer转置，使其形状变为 [K x U]
            #对于TinySecAgg方案，需要在beta点处解码
            
            #TinySecAgg方案——
            U_dec = np.zeros((M, U), dtype=np.int64)
            for m in range(M):
                for j in range(U):
                    numerator = 1
                    denominator = 1
                    for k in range(U):
                        if k != j:
                            numerator = (numerator * (beta[m] - alpha_s_eval[k])) % p
                            denominator = (denominator * (alpha_s_eval[j] - alpha_s_eval[k])) % p
                    inv_denominator = pow(denominator, p-2, p)
                    U_dec[m,j] = (numerator * inv_denominator) % p

            f_recon=np.mod(U_dec.dot(f_eval.T), p)

            t_dec = time.time() - t0_dec  #记录解码阶时间结束
            t_total = time.time() - t_total_start  # 计算总时间
        
        #收集N个用户的时间信息，累加后除以N，得到平均用户时间
            time_users = np.zeros(2, dtype='float')  
            for i in range(N):
                rx_rank = i + 1  #1~N
                tmp = np.empty(2, dtype='float')  #初始化tmp数组，tmp=[0.0, 0.0]
                comm.Recv(tmp, source=rx_rank)  #从rx_rank进程接收用户时间数据，存储在tmp中

                time_users += tmp  #累加
            time_users = time_users / N  #除以N，计算平均用户时间

            # time_set = np.array([t_offline_enc, t_offline_total])

            #记录时间信息：[总时间、用户离线编码时间、用户离线总时间、通信时间、解码时间]
            time_set = np.array([t_total, time_users[0], time_users[1], t_comm, t_dec])

            logging.info('%d-th trial time info=%s' % (avg_idx, time_set))
            #avg_idx——第avg_idx次试验
            time_avg += time_set #累加，后面取平均

            # print avg_idx,'-th trial, # drop users =',np.sum(drop_info)
            # print avg_idx,'-th trial, time=',time_set
            # print
            if avg_idx == n_trail-1:  #若是最后一次试验，同步所有进程
                comm.Barrier()


    
       #拉格朗日插值恢复聚合梯度
        #aggregated_gradient=np.zeros(len(alpha),dtype=np.int64)
        #for j in range(len(alpha)):
         #   basis=np.array([compute_lagrange_basis(alpha[j],beta,n,p) for n in range(M+T)])
          #  aggregated_gradient[j]=(np.dot(aggregated_phi[j],basis)+np.dot(aggregated_psi[j],basis))%p

        elif rank<=N:#User
            comm.Barrier()
            t_offline_start=time.time()
        #offline
        #接收来自Server的公共参数alpha和beta
            alpha=np.empty(N,dtype=np.int64)
            beta=np.empty(M+T,dtype=np.int64)
            comm.Recv(alpha,source=0)
            comm.Recv(beta,source=0)
            
            logging.info("round_idx=%d received alpha and beta "%(avg_idx))
            

            #comm.Recv(alpha,source=0)
            #comm.Recv(beta,source=0)

        #用户生成随机二进制掩码bt_i(大小为d*1，其中随机选择K个元素的值为1，其余元素值为0），用于rand—k稀疏化
        #在0～d-1内随机选择K个不重复的整数
            random_indices=np.random.choice(d,K,replace=False)#随机选择K个坐标
            bt_i=np.zeros(d,dtype=int)
            bt_i[random_indices]=1
            logging.info("round_idx=%d produced bt_i "%(avg_idx))
        #生成k个随机掩码，用于隐藏梯度值——已在generate_lagrange_polynomials中获得
            rt_ik=np.random.randint(0,p,size=K,dtype=np.int64)

        #随机向量v_ikn和u_ikn都为K*T的矩阵，意味着每个选中的K个参数对应T个冗余值，每个冗余值对应一个冗余分片（共M+T个分片中的后T个）（T为用户合谋最大值，因此需要T个冗余分片来确保隐私）
            v_ikn=np.random.randint(0,p,size=(K,T),dtype=np.int64)
            u_ikn=np.random.randint(0,p,size=(K,T),dtype=np.int64)

            #准备数据进行LCC编码
            #对每个被选中的坐标，创建一个单位向量，其中该坐标的值为1，其余坐标的值为0
            X_sub = np.zeros((K + T, 1, d), dtype=np.int64)
            for k in range(K):
                coord = random_indices[k]
                a_k = np.zeros(d, dtype=np.int64)
                a_k[coord] = 1
                X_sub[k, 0, :] = a_k
            #添加随机向量以增加冗余
            for t_idx in range(T):
                X_sub[K + t_idx, 0, :] = np.random.randint(0, p, size=d, dtype=np.int64)

             #使用LCC_encoding_w_Random_partial（）函数进行编码
             # 重塑X_sub以匹配LCC_encoding_w_Random_partial函数的输入要求：X_sub 大小为 [K+T x m x d]
             #指定哪些用户将收到份额
            worker_idx=np.arrange(N)
            X_LCC = LCC_encoding_w_Random_partial(X_sub[:, 0, :], X_sub[K:, 0, :], N, K, T, p, worker_idx)
            
            #对每个用户j，phi_coeff[k,j] = X_LCC[j, 0, random_indices[k]]
            phi_coeff=np.zeros((K,N),dtype=np.int64) #φ_ik(α_j) 的值
            psi_coeff=np.zeros((K,N),dtype=np.int64) #ψ_ik(α_j) 的值
            logging.info("round_idx=%d initial phi_coeff and psi_coeff "%(avg_idx))
            for k in range(K):
                for j in range(N):
                    phi_coeff[k,j] = X_LCC[j, 0, random_indices[k]]
                    psi_coeff[k,j] = (phi_coeff[k,j] * rt_ik[k]) % p
            
            logging.info("round_idx=%d get phi_coeff and psi_coeff "%(avg_idx))
            t_offline_enc = time.time() - t_offline_start
        #用户i将编码后的向量φ_ik(α_j)和ψ_ik(α_j) 发送给其他用户j——实现去中心化的隐私保护，等待在线阶段时使用，确保全局模型内局部梯度参数的正确匹配，同时防止服务器获得对坐标的显式访问
            tx_req1=[None]*(N-1) #存储发送请求的列表
            tx_req2=[None]*(N-1)
            tx_dest=np.delete(range(N),rank-1)
        
            for j in range(len(tx_dest)):
                bf_addr=tx_dest[j]  #目标用户索引，从0开始
                tx_rank=tx_dest[j]+1  #目标用户进程，从1开始
                tx_data1=phi_coeff[bf_addr,:].astype(int)
                tx_data2=psi_coeff[bf_addr,:].astype(int)
                logging.info('[rank=%d] Send phi_coeff and psi_coeff to rank=%d'%(rank, tx_rank))
                tx_req1[j]=comm.Isend([tx_data1,MPI.INT],dest=tx_rank)
                tx_req2[j]=comm.Isend([tx_data2,MPI.INT],dest=tx_rank) #传递K*N矩阵参数的某一行
                    
            if is_sleep: #模拟通信延迟
                comm_time_per_tx=2*N/ comm_mbps / (2**20) * 32
                comm_time = (N-1)* comm_time_per_tx
                time.sleep(comm_time)  #模拟通信延迟，暂停当前进程comm_time秒
            
            #从其他用户接收编码后的向量φ_ik(α_j)和ψ_ik(α_j)
            rx_req=[None]*(N-1) #存储接收请求的列表
            rx_req2=[None]*(N-1)
            rx_source=np.delete(range(N),rank-1)#排除自己
            for i in range(len(rx_source)):
                bf_addr=rx_source[i]
                rx_rank=rx_source[i]+1
                rx_req[i]=comm.Irecv([phi_coeff[bf_addr,:],MPI.INT],source=rx_rank)
                rx_req2[i]=comm.Irecv([psi_coeff[bf_addr,:],MPI.INT],source=rx_rank)

            logging.info('round_idx=%d [rank=%d] is Waiting for phi_coeff and psi_coeff '%(avg_idx,rank))
            MPI.Request.Waitall(tx_req1)
            MPI.Request.Waitall(tx_req2)
            MPI.Request.Waitall(rx_req)
            MPI.Request.Waitall(rx_req2)
            t_offline_total=time.time()-t_offline_start
             
        #online phase round1
            time.sleep(5) #模拟训练时间

            comm.Barrier() #同步，确保所有进程都已经训练完毕，准备开始发送掩码模型
            t0_round1 = time.time() #记录第一轮在线阶段时间开始
            start_encoding_time = time.time() #记录编码开始时间
            local_gradient = np.random.randn(d).satype(np.int64) #模拟本地梯度
        #梯度稀疏化
            sparse_gradient = bt_i*local_gradient #通过二进制掩码获取稀疏化梯度（仅K个非零向量）

        #转换到有限域
            sparse_gradient_in_fp = sparse_gradient % p #将稀疏化梯度转换为有限域
            masked_gradient=np.zeros(K,dtype=np.int64)
            for k in range(K):
                masked_gradient[k] = (sparse_gradient_in_fp[random_indices[k]] - rt_ik[k]) % p #将稀疏化梯度与随机掩码相减，以隐藏梯度值

        #发送掩码梯度给Server
            my_idx=rank-1 #用户索引
            if my_idx in surviving_set1:
                masked_gradient = np.mod(masked_gradient,p).astype(int) #取模避免溢出
                logging.info("(%d) send masked_gradient to Server" % my_idx)
                comm.Send([masked_gradient,MPI.INT], dest=0)  # 发送给服务器
                #logging.info(f"User {current_user_id} sent masked_gradient to the Server (size: {K}).")

            if is_sleep: #模拟通信延迟
                comm_time=K/ comm_mbps / (2**20) * 32
                #comm_time = len(surviving_set1)* comm_time_per_tx #并行发送，无需计算N次
                time.sleep(comm_time)

            t_round1 = time.time() - t0_round1  #计算第一轮在线阶段总耗时

        #online phase round2 #计算编码梯度的本地聚合phi_alpha_i(一个值，待修改)，并发送给Server
            phi_alpha_i = np.zeros(K, dtype=np.int64)
            for k in range(K):
                for j in range(N):
                    phi_alpha_i[k] += (phi_coeff[k,j] * masked_gradient[k] + psi_coeff[k,j])
                    phi_alpha_i[k] = phi_alpha_i[k] % p
                phi_alpha_i[k] = phi_alpha_i[k] % p
            #phi_alpha_i[k] = phi_alpha_i[k] % p

        #发送phi_alpha_j给Server
            if my_idx in surviving_set2:
                logging.info("(%d) send phi_alpha_i to Server" % my_idx)
                comm.Send([phi_alpha_i,MPI.INT], dest=0)  # 并行发送给服务器

            if is_sleep: #模拟通信延迟
                data_size=K
                comm_time=data_size/ comm_mbps / (2**20) * 32
                time.sleep(comm_time)  #模拟通信延迟，暂停当前进程comm_time秒
     

        # 发送时间信息到Server：[离线编码时间, 离线总时间]
            time_set = np.array([t_offline_enc, t_offline_total])
            # comm.send(np.prod(time_set.shape), dest=0)
            comm.Send(time_set, dest=0)  #发送时间信息到Server

          #auto  comm.Barrier()  #等待所有进程完成
          #auto  #记录日志，表示当前进程rank的在线阶段第一轮和第二轮已完成

            if avg_idx == n_trail-1:  #若为最后一次试验，同步所有进程
                comm.Barrier()

#处理和保存结果
    if rank == 0:  #若为Server，记录日志，表示即将显示实验结果
        logging.info("experimental results:")
        time_avg = time_avg / n_trail  #计算平均时间
        logging.info(time_avg)
        logging.info('total running time (sec) = %s' % str(time_avg[0]))  #总运行时间
        logging.info('time for offline = %s' % str(time_avg[2]))  #用户离线总时间

        #保存结果
        result_set = {'N': N,
                      'U': U,
                      'time_set': time_avg
                      }

        time_out.append(result_set) #将result_set添加到time_out列表中，用于存储所有实验结果
        #result_set——单次试验的结果，不同试验次数下的值不同；只包含一组数据
        #time_out——所有试验次数下的结果，包含n_trail组数据
        logging.info('N,U,T= %d, %d, %d'% (N, U, T))
        logging.info('#############################################################################')

#保存所有试验结果
    if rank == 0:  #若为Server进程，将time_out列表序列化为Pickle格式，并保存到文件中，文件名格式为Light_N{N}_d{d}
        pickle.dump(time_out, open('./Light_N' + str(N) + '_d' + str(d), 'wb'), -1)


    




