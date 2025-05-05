#MPI——分布式计算，可能设计多方计算(MPC)或安全聚合(SA)
import logging
import pickle as pickle  #导入pickle模块，用于序列化和反序列化Python对象
import sys  #导入系统模块——用于处理命令行参数和系统退出
import time  #导入时间模块——用于记录时间或控制程序运行速度

#序列化——将一个python对象(如列表、字典、类实例等)转换为一种可以存储或传输的格式(文本形式json、xml、二进制形式pickle等)字节流，以便存储到文件或通过网络传输
#反序列化——与上述过程相反，将序列化后的数据重新转换回原来的Python对象

import numpy as np #用于高效处理数组和矩阵运算
from mpi4py import MPI  #导入MPI模块，用于实现进程间通信，支持分布式计算
#导入常见模块，用于日志记录、数据处理和进程间通信
#导入自定义模块LCC模块，用于轻量级编码和解码
#从sec_agg.mpc_function模块中导入与多方计算MPC相关的函数，用于轻量级编码和解码
from sec_agg.mpc_function import LCC_decoding_with_points, LCC_encoding_with_points # This should now work

#初始化MPI环境
comm = MPI.COMM_WORLD #创建MPI通信器
rank = comm.Get_rank()  #获取当前进程的rank（0为服务器，1~N为用户）
size = comm.Get_size()  #获取总进程数
#进行logging配置，设置日志级别为DEBUG，并设置日志格式(包含进程编号、日期格式、模块名、行号、函数名和日志消息)和日期格式——有助于在运行时跟踪程序的执行情况，方便调试
logging.basicConfig(level=logging.DEBUG,
                    format='%(process)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s',
                    datefmt='%Y-%m-%d,%H:%M:%S')
#sys.argv——获取命令行参数(根据命令行参数数量，设置不同变量)
#N——用户数量；d——特征维度；is_sleep——是否启用睡眠模式(可能用于模拟网络延迟)；
# comm_mbps——网络带宽(单位：Mbps)
#若参数数量不符合要求，主进程（rank==0）会记录错误信息并退出主程序
#服务器节点：rank 0 ; 客户端节点：rank 1~N
#sys.argv[0]——脚本名称；后续元素——用户传入的参数；例如：如果运行 python script.py arg1 arg2，那么 sys.argv 将是 ['script.py', 'arg1', 'arg2']
#灵活处理不同数量的命令行参数，根据参数设置不同的变量值

#处理命令行参数
if len(sys.argv) == 1: #没有提供参数，报错并退出
    if rank == 0:   
        logging.info("ERROR: please input the number of workers")
    exit()
elif len(sys.argv) == 2:
    N = int(sys.argv[1]) #提供了工作节点数量N；int——直接截断取整
    d = 11511784 #模型参数维度，默认约1150万
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

#设置系统参数
#T——容错阈值(设为节点数的一半并向下取整)；
T = int(np.floor(N / 2))

#设置U值数组，U表示参与聚合的用户数量————活跃用户数(T+1)
#U_array = np.array([T + 1, np.floor(0.7 * N), np.floor(0.9 * N)]).astype('int')
U_array = np.array([T + 1]).astype('int')  #U_array=[T+1]
logging.info(U_array) #记录U_array的值

# 设置系统参数
#有限域大小，使用梅森素数，用于加密或模运算（大质数）
p = 2 ** 31 - 1  
#试验重复次数，用于计算平均值
n_trial = 3  # average 100 trials 

#初始化模型，全1向量，用于模拟当前进程模型参数
# my_model = np.random.randint(0,p,size=(d))；默认d = 11511784，约1150万
my_model = np.ones(shape=(d)).astype(int)  

num_pk_per_user = 2  #每个用户的公钥数量

#存储试验结果
time_out = [] 

#处理命令行参数；根据参数数量，设置不同变量，如N、d、is_sleep、comm_mbps等(不同情况下设置的值可能不同)
#MPI(message passing interface)——用于并行计算的通信协议，允许不用进程(rank)之间进行通信和同步

#Barrier——在MPI中为一个同步点，即：所有进程必须到达这个点之后才能继续执行(可确保Barrier之后的所有操作都是在所有进程完成Barrier之前的操作之后进行的)
#在训练模型之前，进程0(服务器)调用Barrier——等待所有客户端进程准备好开始训练
#在接收数据之前，Server再次调用Barrier，确保所有客户端已经发送了数据
#在最后，当avg_n_trail-1时，Server调用Barrier——等待所有客户端完成最后一次试验，最后才能结束整个进程

#Barrier——确保所有进程之间的协调，避免了数据竞争和不一致的情况，这对于并行计算中的正确性和效率至关重要
#不同进程之间的通信——不同进程之间交换数据或信息的过程(分布式系统中，多个进程运行在不同计算节点：不同User的CPU核心或Server)上，需要通过某种方式交换数据，以便完成共同的任务
#通信方式：1.消息传递——进程通过发送和接收消息进行通信（MPI——一种广泛使用的消息传递标准）
#2.共享内存——多个进程共享同一块内存空间，可以直接读写其中的数据（如OpenMP——一种用于多线程编程的共享内存并行编程模型）——通常用于多核处理器或多线程变成，但需要处理内存访问的冲突问题
#3.文件系统——进程通过读写共享文件进行通信——一种间接的通信方式
#通信特点——单向性（一个进程发送消息，另一个进程接收消息）、异步性（发送进程无需等待接收进程的确认，就可以继续执行）、同步性（发送进程需要等待接收进程的确认）、可靠性（通信过程中需要处理错误和异常情况，确保数据传输的准确性和完整性）

#第一部分——初始化与循环
#对每个U值进行试验
#若U_array包含多个值，则循环次数会相应增加
for t in range(len(U_array)):  #len(U_array)=1——表示只有1中活跃用户数量的设置；
    #U是从U_array中提取的值——表示当前试验中参与聚合的活跃用户数量T+1;
    U = np.max([U_array[t], T+1]) # 确保 U-T >= 1，这是LCC编码的要求
    #U的值为U_array中的元素和T+1中的最大值
    d = int(np.ceil(float(d)/(U-T))) * (U-T)  #计算每个用户需要发送的特征维度，调整维度d的大小，确保d可以被(U-T)整除(向上)
    #假设U=10,T=5,d=7;则U-T=5;(float(d)/(U-T))=1.4;np.ceil(float(d)/(U-T))=2;int(np.ceil(float(d)/(U-T)))=2;2*(U-T)=10;最终d=10
    U1 = N #用户总数
    surviving_set1 = np.array(range(U1))  #第一轮存活的节点N

    surviving_set2 = np.array(range(U)) #第二轮存活的节点U

    #设置编码点——确保Server可以正确地恢复聚合结果
    alpha_s = np.array(range(N)) + 1  #用户编码点1~N——用于生成拉格朗日系数矩阵，确保每个用户的编码数据可以被正确地聚合和解码
    beta_s = np.array(range(U)) + 1 #服务器编码点1~U——服务器使用这些编码点来解码接收到的数据，恢复原始的聚合模型

    #初始化时间记录数组：[总时间，用户离线编码时间，用户离线总时间，通信时间，解码时间]
    time_avg = np.zeros((5), dtype=float)  # [t_offline, ] 

#第二部分——日志记录与屏障同步
    if rank == 0: #若为Server，输出开始，执行后续操作
        logging.info("N,U,T=',N,U,T, 'starts!! ")

#第三部分——分布式计算与通信
    #多次试验取平均；n_trail——试验次数，默认为3
    for avg_idx in range(n_trial): 
        comm.Barrier() #同步所有进程
        ##########################################
        ##             服务器端代码              ##
        ##########################################
        if rank == 0:  # 服务器
            # Round 0: offline phase

            comm.Barrier() #同步所有进程——确保所有User准备好开始训练

            t0 = time.time() #记录总时间开始

            #########################################
            # Round 1. online phase, Rx masked model
            #第一轮：接收掩码模型
            # 1.0. Training the model

            # 1.1. Rx the masked model from U users

            comm.Barrier() # 确保所有用户都准备好了开始发送数据，Server也准备好了开始接收数据，保证数据接收的顺序和完整性

            t0_comm = time.time()  #记录通信时间开始
            #初始化接收缓冲区N*d
            x_tilde_buffer = np.zeros((U1, d), dtype=int) 

            #从所有存活用户接收掩码模型（第一轮存活用户N；第二轮存活用户U）
            rx_req = [None] * (U1) #存储MPI的接收请求（长度N，初始化为0）
            for i in surviving_set1: #遍历i=0~N-1
                rx_rank = i + 1  #1~N——接收到的数据的进程编号
                logging.info("rx_rank = %d" % rx_rank)  #表示正在接收来自rx_rank进程的数据
                rx_req[i] = comm.Irecv([x_tilde_buffer[i, :], MPI.INT], source=rx_rank)
                #rx_rank为索引，将数据存储在x_tilde_buffer[i, :]中，类型为MPI.INT

            MPI.Request.Waitall(rx_req) #等待所有接收请求完成
            
            #计算掩码模型的和——将所有用户的掩码模型聚合为一个；对x_tilde_buffer按行求和，然后对结果取模(模数为p)，得到x_tilde_sum
            x_tilde_sum = np.mod(np.sum(x_tilde_buffer, axis=0), p)  #第1~d列的值求和，最终得到d*1的矩阵

            # print x_tilde_buffer
            # print x_sum

            ##########################################################
            # Round 2. online phase, Rx z_tilde and reconstruct sum(z_i) 
            #第二轮：接收z_tilde并重构sum(z_i)

#第4部分——接收与处理z_tilde_buffer
            #初始化接收缓冲区；U*d/(U-T)的零矩阵
            z_tilde_buffer = np.zeros((U, (d / (U - T)).astype(np.int64)), dtype=int)

            #从存活用户接收z_tilde
            rx_req = [None] * (U) #存储MPI的接收请求(长度为U)
            for i in surviving_set2:  #遍历i=0-U-1
                rx_rank = i + 1  #1~U——接收到的数据进程编号
                rx_req[i] = comm.Irecv([z_tilde_buffer[i, :], MPI.INT], source=rx_rank)
                #rx_rank为索引，将数据存储在z_tilde_buffer[i, :]中，类型为MPI.INT
            MPI.Request.Waitall(rx_req) #等待所有接收请求完成

            t_comm = time.time() - t0_comm #记录通信时间结束

    #surviving_set1——第一轮存活节点数N
    #surviving_set2——第二轮存活节点数U
    #x_tilde_buffer——第一轮接收到的掩码模型
    #z_tilde_buffer——第二轮接收到的z_tilde

#第5部分——数据处理与时间记录
            #解码过程
            t0_dec = time.time() #记录解码时间开始
 
            #使用LCC解码重构z
            target_points = alpha_s[surviving_set1] #从alpha_s中提取 surviving_set1 对应的元素，作为要解码的目标点
            #z_recon——解码z_tilde_buffer后得到的掩码数据
            z_recon = LCC_decoding_with_points(z_tilde_buffer, beta_s, target_points, p)
            #调用LCC_decoding_with_points函数，对z_tilde_buffer进行解码，得到z_recon
            
            #计算z的和——将z_tilde_buffer的前U-T行（即幸存U个活跃用户）的数据重塑为(d)的一维数组，得到z_sum
#或许有错误  z_sum = np.reshape(z_tilde_buffer[0:U - T, :], (d,))
            z_sum = np.reshape(z_recon[0:U - T, :], (d,))
            #计算原始模型的和：x_sum = x_tilde_sum - z_sum
            #x_i——用户的真实模型参数，z_i——随机生成的掩码，masked_model = x_i + z_i；
            #Server接收到所有的masked_model后，将它们相加得到x_tilde_sum；
            #z_sum——是通过解码 z_tilde_buffer 得到的，它代表了所有用户掩码的总和；
            #因此，二者相减可以得到原始模型的和x_sum
            x_sum = x_tilde_sum - z_sum #第一轮掩码模型的和-第二轮z_tilde的和
            t_dec = time.time() - t0_dec  #记录解码阶时间结束
            # print x_sum

            t_total = time.time() - t0  #记录总时间结束

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
            if avg_idx == n_trial-1:  #若是最后一次试验，同步所有进程
                comm.Barrier()

        ##########################################
        ##              用户端代码              ##
        ##########################################
        elif rank <= N:  # 用户
            comm.Barrier() #同步所有进程——确保所有用户已经准备好开始进入离线阶段

            #########################
            # Round 0. offline phase
            t0_offline = time.time()  #记录离线阶段开始时间

            # print '[ rank= ',rank,'] Offline start!'
            #生成随机整数掩码z_i；大小d*1，元素范围[0, p)
            z_i = np.random.randint(p, size=(d, 1))

            #生成随机整数噪声n_i；大小(T*d/(U-T))*1，元素范围[0, p)——补充数据，确保数据结构符合编码要求
            n_i = np.random.randint(p, size=((T * d / (U - T)).astype(np.int64), 1))

            #z_i——列向量(大小为d)；n_i——列向量(大小为(T*d/(U-T))*1)；LCC_in——列向量，上面为z_i，下面为n_i(大小为d+(T*d/(U-T)))
            LCC_in = np.concatenate([z_i, n_i], axis=0)
            #重塑LCC_in,大小为U*d/(U-T)
            LCC_in = np.reshape(LCC_in, (U, (d / (U - T)).astype(np.int64)))

            # print rank, 'LCC-in shpae=',np.shape(LCC_in)
            #使用LCC编码函数对LCC_in编码，生成编码后的数据z_tilde_i，大小为U*d/(U-T)
            #每个User独立生成的编码数据，一共有N个用户，每个用户生成一个z_tilde_i
            z_tilde_i = LCC_encoding_with_points(LCC_in, beta_s, alpha_s, p).astype('int')

            t_offline_enc = time.time() - t0_offline  #记录（离线）编码时间

            # z_recon1 = LCC_decoding_with_points(z_tilde_i, alpha_s, beta_s, p)
            # print 'recond:', z_recon1
            # print rank, 'z_tilde_i shpae=',np.shape(z_tilde_i)
            #创建z_tlide_buffer；大小为N*(d/(U-T))——存储所有用户的编码数据z_tilde_i的一部分（有稀疏化操作）
            z_tilde_buffer = np.zeros(shape=(N, (d / (U - T)).astype(np.int64)), dtype=int)
            #每个用户的编码数据 z_tilde_i 的第rank-1行被存入 z_tilde_buffer 的第rank-1行
            #将所有独立用户生成的部分编码数据存储到一个全局缓冲区，方便统一存储与共享
            z_tilde_buffer[rank - 1, :] = z_tilde_i[rank - 1, :]

            #向其他用户发送z_LCC
            tx_req = [None] * (N - 1) #存储MPI的发送请求(长度为N-1)——用于等待所有发送操作完成
            tx_dest = np.delete(range(N), rank - 1)  #目标用户进程编号的列表（排除自己）
            #假设N=4,rank=2,则tx_req=[None, None, None],tx_dest=[0,2,3]，tx_rank=[1,3,4]
            for j in range(len(tx_dest)): #遍历N-1次，为每个目标用户进程创建发送请求
                bf_addr = tx_dest[j]  #目标用户的索引(从0开始)
                tx_rank = tx_dest[j] + 1  #目标用户的进程编号（从1开始；0为Server）
#rank=2生成的编码数据z_tilde_i，其第1行数据被发送到进程1，第3行数据被发送到进程3，第4行数据被发送到进程4；第2行数据则被存入z_tilde_buffer的第2行中
                tx_data = z_tilde_i[bf_addr, :].astype(int)  #提取当前目标用户的编码数据tx_data，确保其为是整数
                logging.info('[ rank= %d ] send to %d' % (rank, tx_rank))  #记录日志，表示当前进程rank正在向目标进程tx_rank发送数据
                tx_req[j] = comm.Isend([tx_data, MPI.INT], dest=tx_rank)  #向目标进程tx_rank发送数据tx_data
                #Isend——启动一个发送操作，并立即返回；tx_data——待发送的数据；MPI.INT——数据类型为int；dest=tx_rank——数据将发送到的目标进程编号
                # tx_req[j]存储发送请求对象
            #模拟通信延迟
            if is_sleep:
                #data_size = len(tx_dest) * d/(U-T) * 32 # bit
                #计算每次发送的通信时间
                comm_time_per_tx = d / (U-T) / comm_mbps / (2**20) * 32#sec
                #总通信时间=目标用户数量*每次发送的时间
                comm_time = len(tx_dest) * comm_time_per_tx 

                time.sleep(comm_time)  #模拟通信延迟，暂停当前进程comm_time秒

            #从其他用户接收z_LCC
            rx_req = [None] * (N - 1)  #存储接收请求列表，长度为N-1
            rx_source = np.delete(range(N), rank - 1) #生成(除当前用户外)需要接收数据的目标用户进程编号的列表

            for i in range(len(rx_source)): #遍历N-1次，为每个目标用户创建接收请求
                bf_addr = rx_source[i]  #目标用户的索引(从0开始)
                rx_rank = rx_source[i] + 1  #目标用户的进程编号（从1开始；0为Server）

                rx_req[i] = comm.Irecv([z_tilde_buffer[bf_addr, :], MPI.INT], source=rx_rank)  #从目标进程rx_rank接收数据，存储在z_tilde_buffer的bf_addr（=rx_rank-1）行
                #Irecv——启动一个接收操作；source=rx_rank——将接收数据的目标进程编号
##rank=2从进程1、进程3、进程4接收数据(进程1、3、4分别生成的编码数据z_tilde_i的第2行)，分别存入z_tilde_buffer的第0行、第2行、第3行1行、第3行、第4行
            logging.info("round_idx = %d [ rank= %d ] waiting..." % (avg_idx, rank))
            #记录日志，表示当前进程rank正在等待接收数据
            MPI.Request.Waitall(tx_req)  #等待所有发送完成
            MPI.Request.Waitall(rx_req)  #等待所有接收完成
            logging.info("round_idx = %d [ rank= %d ] finish send/recv." % (avg_idx, rank))
            #记录日志，表示当前进程rank的发送/接收操作已完成

            t_offline_total = time.time() - t0_offline  #记录离线阶段总时间

            ########################################################
            # Round 1. online phase, send x_i + z_i to the server
            #在线阶段第一轮：发送掩码模型到Server
            # 1.0. Training the model (normally cost more than 10 seconds)
            #模拟模型训练(通常需要10秒以上)
            x_i = rank * np.ones((d, 1), dtype=int) #简化的模型，实际应该是训练结果
            time.sleep(5) #模拟训练时间

            comm.Barrier() #同步，确保所有进程都已经训练完毕，准备开始发送掩码模型
            t0_round1 = time.time() #记录第一轮在线阶段时间开始

            #发送掩码模型到服务器
            # 1.1. Sending the masked model to the server
            my_idx = rank - 1 #将进程编号（从1开始）转换为索引（从0开始）
            if my_idx in surviving_set1:  #若当前索引为第一轮中存活索引之一，则计算并发送掩码模型给Server
                masked_model = np.mod(x_i + z_i, p).astype(int)  #计算掩码模型：x_i+z_i
                logging.info("(%d)send masked_model" % my_idx)  #记录日志，表示当前用户正在发送掩码模型
                comm.Isend([masked_model, MPI.INT], dest=0)  #dest=0表示发送给Server

            #模拟通信延迟
            if is_sleep:
                data_size = d # 参数数量
                comm_time = data_size / comm_mbps / (2**20) * 32 #秒
                time.sleep(comm_time)  #让当前进程暂停comm_time秒，模拟通信延迟

            t_round1 = time.time() - t0_round1  #计算第一轮在线阶段总耗时

            ########################################################
            # Round 2. online phase, send z_tilde to the server
            #在线阶段第二轮：发送z_tilde到Server
            #计算z_tilde的和
            #z_tilde_buffer——之前生成的编码数据；从z_tilde_buffer中提取surviving_set1对应的行，计算每列的和，然后对结果取模p，并转换为整数类型
            z_tilde_sum = np.mod(np.sum(z_tilde_buffer[surviving_set1, :], axis=0), p).astype('int')

            if my_idx in surviving_set2: #若当前索引为第二轮中存活索引之一，则计算并发送z_tilde_sum给Server
                logging.info("(%d)send z_tilde_sum" % my_idx)
                comm.Isend([z_tilde_sum, MPI.INT], dest=0) #异步发送数据到Server
            #模拟通信延迟
            if is_sleep:
                data_size = d/(U-T)  # 参数数量/数据大小
                comm_time = data_size / comm_mbps / (2**20) * 32 #秒 #计算通信时间
                time.sleep(comm_time)  #让当前进程暂停comm_time秒，模拟通信延迟

          #auto  t_round2 = time.time() - t0_round1  #计算第二轮在线阶段总耗时

          #auto  t_offline_enc = time.time() - t0_offline  #记录离线编码时间

            # 发送时间信息到Server：[离线编码时间, 离线总时间]
            time_set = np.array([t_offline_enc, t_offline_total])
            # comm.send(np.prod(time_set.shape), dest=0)
            comm.Send(time_set, dest=0)  #发送时间信息到Server

          #auto  comm.Barrier()  #等待所有进程完成
          #auto  #记录日志，表示当前进程rank的在线阶段第一轮和第二轮已完成

            if avg_idx == n_trial-1:  #若为最后一次试验，同步所有进程
                comm.Barrier()
    #处理和保存结果
    if rank == 0:  #若为Server，记录日志，表示即将显示实验结果
        logging.info("experimental results:")
        time_avg = time_avg / n_trial  #计算平均时间
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
    # Indent the following lines to be inside the if block
    filename = f'Light_N{N}_d{d}.pkl' # Construct the filename
    with open(filename, 'wb') as f: # Open the file in binary write mode
        pickle.dump(time_out, f) # Serialize and save the time_out list
    logging.info(f"Results saved to {filename}") # Log the save operation
