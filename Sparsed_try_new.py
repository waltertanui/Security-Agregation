import numpy as np
from mpi4py import MPI
import logging
import math
import pickle
import time


# 设置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# 系统参数
p = 2**61 - 1  # 有限域大小
N = 100        # 用户数量
d = 1000000     # 模型参数维度
K = int(0.01*d)     # 稀疏化参数，每个用户只发送K个参数
M = 40          # 分片数量
T = int(np.floor(N/2) )  # 容错参数

# 初始化MPI环境
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size != N + 1:
    raise ValueError("Need 1 server and N users")

def compute_lagrange_basis(alpha, beta, n, p): #传入当前分片索引n
    numerator = 1
    denominator = 1
    for m in range(len(beta)):
        if m != n:
            numerator = (numerator * (alpha - beta[m])) % p
            denominator = (denominator * (beta[n] - beta[m])) % p
    inv_denominator = pow(denominator, p-2, p)
    return (numerator * inv_denominator) % p

#离线阶段
def offline_stage(comm, rank, N, M, T, d, K, p):
    """
    TinySecAgg离线阶段：生成公共参数、随机掩码、拉格朗日多项式系数
    """
    if rank == 0:
        # 服务器生成全局公共参数 α (N个) 和 β (M+T个)
        alpha = np.random.randint(1, p, size=N, dtype=np.int64)   # α_i ∈ Fp, i=1..N
        beta = np.random.randint(1, p, size=M+T, dtype=np.int64)  # β_n ∈ Fp, n=1..M+T
        # 广播参数给所有用户
        for user_idx in range(1, N+1):
            comm.Send(alpha, dest=user_idx)
            comm.Send(beta, dest=user_idx)
    else:
        # 普通用户接收公共参数
        alpha = np.empty(N, dtype=np.int64)
        beta = np.empty(M+T, dtype=np.int64)
        comm.Recv(alpha, source=0)
        comm.Recv(beta, source=0)
    
    # 用户生成随机二进制掩码（rand-K稀疏化）
    
    random_indices = np.random.choice(d, K, replace=False)  # 随机选择K个坐标
    bt_i = np.zeros(d, dtype=int)
    bt_i[random_indices] = 1  # 二进制掩码，选中坐标置1
    
    # 生成随机掩码和冗余参数
    rt_ik = np.random.randint(0, p, size=K, dtype=np.int64)  # K个随机掩码，用于隐藏梯度值
    v_ikn = np.random.randint(0, p, size=(K, T), dtype=np.int64)  # 冗余分片随机值（后T个β）
    u_ikn = np.random.randint(0, p, size=(K, T), dtype=np.int64)
    
    # 构造拉格朗日插值多项式系数
    phi_coeff = []  # φ_ik(α_j) 的值，存储为K个参数的多项式在α_j处的值
    psi_coeff = []  # ψ_ik(α_j) 的值
    
    for k in range(K):
        coord = random_indices[k]  # 当前选中的坐标
        a_k = np.zeros(d, dtype=int)
        a_k[coord] = 1  # 单位向量，仅选中坐标为1
        a_k_shards = np.array_split(a_k, M)  # 分割为M个等长分片，每片大小d/M
        
        phi = np.zeros(N, dtype=np.int64)  # 每个用户j对应α_j的φ值
        psi = np.zeros(N, dtype=np.int64)
        
        for j in range(N):  # 对每个用户j，计算φ_ik(α_j)和ψ_ik(α_j)
            alpha_j = alpha[j]
            basis_sum_phi = 0
            basis_sum_psi = 0
            
            # 前M项：有效分片（真实数据）
            for n in range(M):
                shard = a_k_shards[n]
                local_idx = coord // (d // M)  # 当前坐标在分片内的索引
                a_kn = shard[local_idx] if local_idx < len(shard) else 0
                basis = compute_lagrange_basis(alpha_j, beta, n, p)
                basis_sum_phi += a_kn * basis
                basis_sum_psi += a_kn * rt_ik[k] * basis
            
            # 后T项：冗余分片（随机值）
            for n in range(M, M+T):
                v = v_ikn[k, n-M]
                u = u_ikn[k, n-M]
                basis = compute_lagrange_basis(alpha_j, beta, n, p)
                basis_sum_phi += v * basis
                basis_sum_psi += u * basis
            
            phi[j] = basis_sum_phi % p
            psi[j] = basis_sum_psi % p
        
        phi_coeff.append(phi)
        psi_coeff.append(psi)
    
    # 用户将phi和psi系数发送给服务器（或其他用户，根据协议设计）
    if rank != 0:
        for j in range(N):
            if j + 1 == rank:  # 假设用户编号从1开始
                continue  # 跳过自己
            comm.Send(phi_coeff, dest=j+1)
            comm.Send(psi_coeff, dest=j+1)
    
    return alpha, beta, random_indices, rt_ik, phi_coeff, psi_coeff

#User在线阶段第一轮——生成掩码梯度并计算聚合值
def online_stage_round1():
    """
    在线阶段第一轮次操作
    """
    # 本地训练得到梯度∆t_i
    local_gradient = np.random.randint(0, p, size=d)
    
    # 稀疏化操作，选择K个参数
    bt_i = np.zeros(d, dtype=int)
    random_indices = np.random.choice(d, K, replace=False)
    bt_i[random_indices] = 1
    xt_i = local_gradient[random_indices]  # 直接使用离线生成的坐标
    
    # 将稀疏化的本地梯度xt_i转换为有限域Fp
    xt_i = xt_i % p
    
    # 广播掩码梯度参数xt_ik = xt_i(Kt_i(k)) − offline_rt_ik[k]
    for k in range(K):
        rt_ik = np.random.randint(0, p)
        xt_ik = xt_i[k] - rt_ik[k]  #使用离线掩码
        xt_ik = xt_ik % p
        
        # 发送到服务器
        comm.Isend([xt_ik, MPI.INT], dest=0)

def online_stage_round2():
    """
    在线阶段第二轮次操作
    """
    # 接收其他用户的编码向量
    phi_ik = np.zeros(N)
    psi_ik = np.zeros(N)
    for j in range(N):
        if j != rank-1:
            phi_ik[j] = comm.Irecv(source=j+1)
            psi_ik[j] = comm.Irecv(source=j+1)
    
    # 计算本地聚合的编码梯度
    local_aggregate = 0
    for j in range(N):
        local_aggregate += phi_ik[j] + psi_ik[j]
    local_aggregate = local_aggregate % p
    
    # 发送到服务器
    comm.Isend([local_aggregate, MPI.INT], dest=0)

def store_encoded_vectors(phi_ik, psi_ik, user_idx):
    """
    存储编码向量
    """
    # 实际应用中，这里应该将编码向量存储到持久化存储中
    logging.info(f"Server: Stored encoded vectors from user {user_idx}")

def online_stage_round1_server():
    """
    服务器端在线阶段第一轮次操作
    """
    logging.info("Server: Online stage round 1 started")
    # 接收掩码梯度
    masked_gradients = []
    for user_idx in range(1, N+1):
        xt_ik = np.empty(1, dtype=np.int64)
        comm.Recv(xt_ik, source=user_idx)
        masked_gradients.append(xt_ik)
    # 聚合掩码梯度
    aggregated_masked_gradient = aggregate_masked_gradients(masked_gradients)
    # 准备聚合结果
    prepare_aggregated_result(aggregated_masked_gradient)
    logging.info("Server: Online stage round 1 completed")

def aggregate_masked_gradients(masked_gradients):
    """
    聚合掩码梯度
    """
    # 实际应用中，这里应该对所有用户的掩码梯度进行聚合
    aggregated_gradient = np.sum(masked_gradients, axis=0) % p
    return aggregated_gradient

def prepare_aggregated_result(aggregated_gradient):
    """
    准备聚合结果
    """
    logging.info("Server: Prepared aggregated result")

def online_stage_round2_server():
    """
    服务器端在线阶段第二轮次操作
    """
    logging.info("Server: Online stage round 2 started")
    # 接收来自M+T个用户的φ(αi)
    phi_alpha_i = []
    for _ in range(M+T):
        phi = np.empty(1, dtype=np.int64)
        comm.Recv(phi)
        phi_alpha_i.append(phi)
    # 通过多项式插值恢复聚合梯度xt_agg
    xt_agg = recover_aggregated_gradient(phi_alpha_i)
    # 更新全局模型
    update_global_model(xt_agg)
    logging.info("Server: Online stage round 2 completed")

def recover_aggregated_gradient(phi_alpha_i, beta, p):
    # phi_alpha_i是M+T个点的值，beta是对应的x坐标
    aggregated = np.zeros(M, dtype=int)
    for m in range(M):
        # 计算拉格朗日基在beta[m]处的值
        basis = 1
        for n_prime in range(M+T):
            if n_prime != m:
                basis = (basis * (beta[m] - beta[n_prime])) % p
        inv_denominator = pow(basis, p-2, p)
        total = 0
        for idx, phi in enumerate(phi_alpha_i):
            term = (phi * inv_denominator) % p
            for n_prime in range(M+T):
                if n_prime != idx:
                    term = (term * (beta[m] - beta[n_prime])) % p
            total = (total + term) % p
        aggregated[m] = total
    return aggregated

def update_global_model(xt_agg):
    """
    更新全局模型
    """
    logging.info(f"Server: Updated global model with aggregated gradient: {xt_agg}")
#Server端不同阶段函数

if __name__ == "__main__":
    if rank == 0:
        # 服务器端操作
        logging.info("服务器启动")
        # 离线阶段
        offline_stage()
        # 在线阶段第一轮次
        online_stage_round1_server()
        # 在线阶段第二轮次
        online_stage_round2_server()
    else:
        # 用户端操作
        print(f"用户{rank}启动...")
        
        # 执行离线阶段
        offline_stage()
        
        # 执行在线阶段第一轮次
        online_stage_round1()
        
        # 执行在线阶段第二轮次
        online_stage_round2()