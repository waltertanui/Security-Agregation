
import logging
import pickle as pickle  
import sys  
import time
import numpy as np 
from mpi4py import MPI

#初始化MPI环境
comm = MPI.COMM_WORLD #创建MPI通信器
rank = comm.Get_rank()  #获取当前进程的rank（0为服务器，1~N为用户）
size = comm.Get_size()  #获取总进程数

#进行logging配置，设置日志级别为DEBUG，并设置日志格式(包含进程编号、日期格式、模块名、行号、函数名和日志消息)和日期格式——有助于在运行时跟踪程序的执行情况，方便调试
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
N = 100 #用户总数

#T——容错阈值(设为节点数的一半并向下取整)；
T = int(np.floor(N / 2))
M = 40 #碎片数量，用于控制通信效率和对抗能力之间的权衡
#设置U值数组，U表示参与聚合的用户数量————活跃用户数(T+1)
U_array = np.array([T + 1]).astype('int')
logging.info(U_array) #记录U_array的值

# 设置系统参数
#有限域大小，使用梅森素数，用于加密或模运算（大质数）
p_field = 2 ** 31 - 1  
#试验重复次数，用于计算平均值
n_trial = 3  # average 100 trials 
#----------the first change: Add sparsity_ratio------------------------------------------------
K = int(0.01*d)  # 稀疏化参数（论文中取0.01d或0.05d）

#初始化模型，全1向量，用于模拟当前进程模型参数
my_model = np.ones(shape=(d)).astype(int)  

num_pk_per_user = 2  #每个用户的公钥数量

#存储试验结果
time_out = [] 

#Server端离线阶段——接收并存储来自每个用户的编码向量
def TinySecAgg_server_offline(N, K, M, T, p_field):
    #服务器端离线阶段
    #初始化存储编码向量的数组
    phi_ik = np.zeros((N, K, M + T), dtype=int)
    psi_ik = np.zeros((N, K, M + T), dtype=int)

    #接收来自每个用户的编码向量
    for user in range(N):
        phi_ik[user] = np.empty((K, M + T), dtype=int)
        psi_ik[user] = np.empty((K, M + T), dtype=int)
        comm.Recv([phi_ik[user], MPI.INT], source=user)
        comm.Recv([psi_ik[user], MPI.INT], source=user)
    logging.info("Server离线阶段完成：接收来自所有用户的编码向量")
    return phi_ik, psi_ik

#Server端在线阶段第一轮次——接收并聚合掩码梯度
def TinySecAgg_server_online_round1(N, d, p_field):

    masked_xt=np.zeros((N, d), dtype=int)
    for user in range(N):
        masked_xt[user] = np.empty(d, dtype=int)
        comm.Recv([masked_xt[user], MPI.INT], source=user)
    
    #聚合来自每个用户的编码向量(聚合掩码梯度)
    aggregated_masked_xt = np.sum(masked_xt, axis=0) % p_field
    logging.info("Server在线阶段第一轮次完成: 聚合掩码梯度")
    return aggregated_masked_xt

#Server端在线阶段第二轮次——接收本地聚合梯度并重建稀疏化梯度
def TinySecAgg_server_online_round2(N, M,d, p_field):

    #初始化存储本地聚合编码梯度的数组
    phi_alpha = np.zeros((N, M + T), dtype=int)
    for user in range(N):
        phi_alpha[user] = np.empty(M + T, dtype=int)
        comm.Recv([phi_alpha[user], MPI.INT], source=user)

    aggregated_xt = np.zeros(M + T, dtype=int)
    for i in range(M + T):
        aggregated_xt[i] = np.sum(phi_alpha[:, i]) % p_field
    
    logging.info("Server在线阶段第二轮次完成: 重建稀疏化梯度")
    return aggregated_xt

#User端离线阶段——生成编码向量并发送给其他用户
def TinySecAgg_user_offline(N, K, M, T, p_field, alpha_s, beta_s):
   
    # 生成随机二进制掩码
    bt_i = np.random.randint(2, size=d)
    Kt_i = np.where(bt_i == 1)[0]  # 选择 K 个参数

    # 生成拉格朗日多项式
    phi_ik = np.zeros((K, M + T), dtype=int)
    psi_ik = np.zeros((K, M + T), dtype=int)
    for k in range(K):
        rt_ik = np.random.randint(p_field)
        vt_ikn = np.random.randint(p_field, size=M + T)
        ut_ikn = np.random.randint(p_field, size=M + T)
        for n in range(M + T):
            # 计算多项式系数
            phi_ik[k, n] = (rt_ik + np.sum(vt_ikn[n+1:])) % p_field
            psi_ik[k, n] = (rt_ik + np.sum(ut_ikn[n+1:])) % p_field

    # 发送编码向量到其他用户
    for j in range(N):
        if j != rank:
            comm.Isend([phi_ik, MPI.INT], dest=j)
            comm.Isend([psi_ik, MPI.INT], dest=j)

    logging.info("User端离线阶段完成: 生成编码向量并发送给其他用户")

#User端在线阶段第一轮次——生成掩码梯度并发送给服务器
def TinySecAgg_user_online_round1(d, p_field):

    # 生成稀疏化梯度
    xt_i = np.random.randint(p_field, size=d)
    
    # 生成随机掩码
    mask = np.random.randint(p_field)
    
    # 生成掩码梯度
    masked_xt_i = np.mod(xt_i + mask, p_field)
    
    # 发送掩码梯度到服务器
    comm.Isend([masked_xt_i, MPI.INT], dest=0)
    
    logging.info("用户端在线阶段第一轮次完成：生成掩码梯度并发送给Server")

#User端在线阶段第二轮次——生成本地聚合编码梯度并发送给服务器
def TinySecAgg_user_online_round2(N, M, T, p_field, phi_ik, psi_ik):

    # 生成本地聚合编码梯度
    phi_alpha_i = np.zeros(M + T, dtype=int)
    for k in range(K):
        for n in range(M + T):
            phi_alpha_i[n] += phi_ik[k, n]
    
    # 发送本地聚合编码梯度到服务器
    comm.Isend([phi_alpha_i, MPI.INT], dest=0)
    
    logging.info("用户端在线阶段第二轮次完成：生成本地聚合编码梯度并发送给服务器")

#第一部分——初始化与循环
#对每个U值进行试验
for t in range(len(U_array)): 
    U = np.max([U_array[t], T+1]) # 确保 U-T >= 1——确保在掉线用户少于N/2的情况下，Server仍能正确聚合梯度

    d = int(np.ceil(float(d)/(U-T))) * (U-T)  #计算每个用户需要发送的特征维度，调整维度d的大小，确保d可以被(U-T)整除(向上)
    #假设U=10,T=5,d=7;则U-T=5;(float(d)/(U-T))=1.4;np.ceil(float(d)/(U-T))=2;int(np.ceil(float(d)/(U-T)))=2;2*(U-T)=10;最终d=10
    U1 = N #用户总数
    surviving_set1 = np.array(range(U1))  #第一轮存活的节点N

    surviving_set2 = np.array(range(U)) #第二轮存活的节点U

    #设置编码点
    alpha_s = np.array(range(N)) + 1  #用户编码点
    beta_s = np.array(range(U)) + 1 #服务器编码点

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

            comm.Barrier() #同步所有进程

            t0 = time.time() #记录总时间开始

            #离线阶段
            #phi_ik, psi_ik = TinySecAgg_server_offline(N, K, M, T, p_field)
            #初始化存储编码向量的数组
            t0_comm = time.time() #记录通信时间开始
            phi_ik = np.zeros((N, K, M + T), dtype=int)
            psi_ik = np.zeros((N, K, M + T), dtype=int)
            #接收来自每个用户的编码向量
            for user in range(N):
                phi_ik[user] = np.empty((K, M + T), dtype=int)
                psi_ik[user] = np.empty((K, M + T), dtype=int)
                comm.Recv([phi_ik[user], MPI.INT], source=user)
                comm.Recv([psi_ik[user], MPI.INT], source=user)
            logging.info("Server离线阶段完成：接收来自所有用户的编码向量")

            #在线阶段第一轮次
            #aggregated_masked_xt = TinySecAgg_server_online_round1(N, d, p_field)
            #初始化掩码模型（User端）


#Server端在线阶段第一轮次——接收并聚合掩码梯度
def TinySecAgg_server_online_round1(N, d, p_field):

    masked_xt=np.zeros((N, d), dtype=int)
    for user in range(N):
        masked_xt[user] = np.empty(d, dtype=int)
        comm.Recv([masked_xt[user], MPI.INT], source=user)
    
    #聚合来自每个用户的编码向量(聚合掩码梯度)
    aggregated_masked_xt = np.sum(masked_xt, axis=0) % p_field
    logging.info("Server在线阶段第一轮次完成: 聚合掩码梯度")
    return aggregated_masked_xt

#Server端在线阶段第二轮次——接收本地聚合梯度并重建稀疏化梯度
def TinySecAgg_server_online_round2(N, M,d, p_field):

    #初始化存储本地聚合编码梯度的数组
    phi_alpha = np.zeros((N, M + T), dtype=int)
    for user in range(N):
        phi_alpha[user] = np.empty(M + T, dtype=int)
        comm.Recv([phi_alpha[user], MPI.INT], source=user)

    aggregated_xt = np.zeros(M + T, dtype=int)
    for i in range(M + T):
        aggregated_xt[i] = np.sum(phi_alpha[:, i]) % p_field
    
    logging.info("Server在线阶段第二轮次完成: 重建稀疏化梯度")
    return aggregated_xt

            # 在线阶段第二轮次
            aggregated_xt = TinySecAgg_server_online_round2(N, M, T, p_field)

            # 更新全局模型
            global_model = aggregated_xt
            logging.info("全局模型已成功更新！")
    
        ##########################################
        ##              用户端代码              ##
        ##########################################
        
        elif rank<=N:
            comm.Barrier()
            t0_offline = time.time() #记录离线阶段开始
            #离线阶段
            #两个二进制向量alpha_s和beta_s
            alpha_s = np.array(range(N)) + 1
            beta_s = np.array(range(M + T)) + 1
            #TinySecAgg_user_offline(N, K, M, T, p_field, alpha_s=alpha_s, beta_s=beta_s)
            # 生成随机二进制掩码，选择 K 个参数进行稀疏化（？rand——K/top——K）
            bt_i = np.random.randint(2, size=d)
            Kt_i = np.where(bt_i == 1)[0]  # 选择 K 个参数
    
    #####################################################################################################

            # 在线阶段第一轮次
            TinySecAgg_user_online_round1(d, p_field)
        
            # 在线阶段第二轮次
            phi_ik = np.random.randint(p_field, size=(K, M + T))
            TinySecAgg_user_online_round2(N, M, T, p_field, phi_ik=phi_ik)

#User端离线阶段——生成编码向量并发送给其他用户
def TinySecAgg_user_offline(N, K, M, T, p_field, alpha_s, beta_s):
   
    # 生成随机二进制掩码
    bt_i = np.random.randint(2, size=d)
    Kt_i = np.where(bt_i == 1)[0]  # 选择 K 个参数

    # 生成拉格朗日多项式
    phi_ik = np.zeros((K, M + T), dtype=int)
    psi_ik = np.zeros((K, M + T), dtype=int)
    for k in range(K):
        rt_ik = np.random.randint(p_field)
        vt_ikn = np.random.randint(p_field, size=M + T)
        ut_ikn = np.random.randint(p_field, size=M + T)
        for n in range(M + T):
            # 计算多项式系数
            phi_ik[k, n] = (rt_ik + np.sum(vt_ikn[n+1:])) % p_field
            psi_ik[k, n] = (rt_ik + np.sum(ut_ikn[n+1:])) % p_field

    # 发送编码向量到其他用户
    for j in range(N):
        if j != rank:
            comm.Isend([phi_ik, MPI.INT], dest=j)
            comm.Isend([psi_ik, MPI.INT], dest=j)

    logging.info("User端离线阶段完成: 生成编码向量并发送给其他用户")

#User端在线阶段第一轮次——生成掩码梯度并发送给服务器
def TinySecAgg_user_online_round1(d, p_field):

    # 生成稀疏化梯度
    xt_i = np.random.randint(p_field, size=d)
    
    # 生成随机掩码
    mask = np.random.randint(p_field)
    
    # 生成掩码梯度
    masked_xt_i = np.mod(xt_i + mask, p_field)
    
    # 发送掩码梯度到服务器
    comm.Isend([masked_xt_i, MPI.INT], dest=0)
    
    logging.info("用户端在线阶段第一轮次完成：生成掩码梯度并发送给Server")

#User端在线阶段第二轮次——生成本地聚合编码梯度并发送给服务器
def TinySecAgg_user_online_round2(N, M, T, p_field, phi_ik, psi_ik):

    # 生成本地聚合编码梯度
    phi_alpha_i = np.zeros(M + T, dtype=int)
    for k in range(K):
        for n in range(M + T):
            phi_alpha_i[n] += phi_ik[k, n]
    
    # 发送本地聚合编码梯度到服务器
    comm.Isend([phi_alpha_i, MPI.INT], dest=0)
    
    logging.info("用户端在线阶段第二轮次完成：生成本地聚合编码梯度并发送给服务器")
