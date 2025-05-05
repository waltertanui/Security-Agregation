

elif rank<=N:#User
            comm.Barrier()
            t_offline_start=time.time()
        #offline
        #接收来自Server的公共参数alpha和beta
            alpha=np.empty(N,dtype=np.int64)
            beta=np.empty(M+T,dtype=np.int64)
            comm.Recv(alpha,source=0)
            comm.Recv(beta,source=0)

        #用户生成随机二进制掩码bt_i(大小为d*1，其中随机选择K个元素的值为1，其余元素值为0），用于rand—k稀疏化
        #在0～d-1内随机选择K个不重复的整数
            random_indices=np.random.choice(d,K,replace=False)#随机选择K个坐标
            bt_i=np.zeros(d,dtype=int)
            bt_i[random_indices]=1
        #生成k个随机掩码，用于隐藏梯度值——已在generate_lagrange_polynomials中获得
        #rt_ik=np.random.randint(0,p,size=K,dtype=np.int64)

        ##v_ikn和u_ikn都为K*T的矩阵，意味着每个选中的K个参数对应T个冗余值，每个冗余值对应一个冗余分片（共M+T个分片中的后T个）（T为用户合谋最大值，因此需要T个冗余分片来确保隐私）
        #v_ikn=np.random.randint(0,p,size=(K,T),dtype=np.int64)
        #u_ikn=np.random.randint(0,p,size=(K,T),dtype=np.int64)

            phi_coeff=np.zeros((K,N),dtype=np.int64) #φ_ik(α_j) 的值
            psi_coeff=np.zeros((K,N),dtype=np.int64) #ψ_ik(α_j) 的值
        
            phi_coeff,psi_coeff,rt_ik,v_ikn,u_ikn = generate_lagrange_polynomials(alpha,beta,M,T,K,random_indices,d,p)
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

                #for j in range(0,N):
                    #if j  == current_user_id:  
                        #continue  # 跳过自己
                    #dest_rank = j 
                    #phi_coeff = phi_coeff % p #取模避免溢出
                    #psi_coeff = psi_coeff % p
                    #comm.Send(phi_coeff, dest=dest_rank)
                    #comm.Send(psi_coeff, dest=dest_rank) #传递完整的K*N矩阵参数
                    #logging.info('[rank=%d] Send phi_coeff and psi_coeff to rank=%d'%(rank, dest_rank))

            #if is_sleep: #模拟通信延迟
                #comm_time_per_tx=2*K*N/ comm_mbps / (2**20) * 32
                #comm_time = (N-1)* comm_time_per_tx 
                #time.sleep(comm_time)  #模拟通信延迟，暂停当前进程comm_time秒

        #从其他用户接收编码后的向量φ_ik(α_j)和ψ_ik(α_j)
            #rx_req=[None]*(N-1) #存储接收请求的列表
            #rx_req2=[None]*(N-1)
            #rx_source=np.delete(range(N),rank-1)#排除自己
            #for i in range(len(rx_source)):
                #bf_addr=rx_source[i]
               # rx_rank=rx_source[i]+1
               # rx_req[i]=comm.Irecv([phi_coeff[i],MPI.INT64_T],source=rx_rank)
               # rx_req2[i]=comm.Irecv([psi_coeff[i],MPI.INT64_T],source=rx_rank)

            logging.info('round_idx=%d [rank=%d] is Waiting for phi_coeff and psi_coeff '%(avg_idx,rank))
            MPI.Request.WaitAll(tx_req1)
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
                masked_gradient[k] = sparse_gradient_in_fp[random_indices[k]] - rt_ik[k] #将稀疏化梯度与随机掩码相减，以隐藏梯度值

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
            phi_alpha_i[k] = phi_alpha_i[k] % p

        #发送phi_alpha_j给Server
            if my_idx in surviving_set2:
                logging.info("(%d) send phi_alpha_i to Server" % my_idx)
                comm.Send([phi_alpha_i,MPI.INT], dest=0)  # 并行发送给服务器

            if is_sleep: #模拟通信延迟
                data_size=K
                comm_time=data_size/ comm_mbps / (2**20) * 32
                time.sleep(comm_time)  #模拟通信延迟，暂停当前进程comm_time秒
     