import numpy as np

np.random.seed(42)  # set the seed of the random number generator for consistency
#设置一个随机数生成器的种子；为了确保结果的一致性，生成的随机数是确定的，即相同的输入会生成相同的随机数

def modular_inv(a, p): #计算a在模p下的乘法逆元，即找到整数x，使得a*x ≡ 1 (mod p)
    x, y, m = 1, 0, p
    while a > 1:
        q = a // m;
        t = m;

        m = np.mod(a, m)
        a = t
        t = y

        y, x = x - np.int64(q) * np.int64(y), t

        if x < 0: #如果x为负数，通过模运算将其转换为正数
            x = np.mod(x, p)
    return np.mod(x, p)
#使用扩展欧几里得算法实现，通过不断迭代更新x和y，直到a变为1，此时x即为a的模逆元

#计算模p下的除法函数，即计算a/b在模p下的结果
def divmod(_num, _den, _p): #计算_num/_den在模p下的值——相当于_num乘以_den的模逆元后的结果在模p下的值
    # compute num / den modulo prime p
    _num = np.mod(_num, _p)#首先将_num和_den分别对p取模，确保它们在模p的范围内
    _den = np.mod(_den, _p)
    _inv = modular_inv(_den, _p) #使用modular_inv函数计算_den的模逆元_inv
    # print(_num,_den,_inv)
    return np.mod(np.int64(_num) * np.int64(_inv), _p) #计算_num乘以_inv，并对p取模，得到结果

#计算输入值列表vals在模p下的乘积——vals中所有值的乘积，并对p取模
def PI(vals, p):  # upper-case PI -- product of inputs
    accum = 1 #初始化累乘器为1
    # print vals
    for v in vals: #对于每个输入值v，先对p取模，然后乘以accum，再对p取模，更新accum
        # print accum, v, np.mod(v,p)
        tmp = np.mod(v, p)
        accum = np.mod(accum * tmp, p)
    return accum #返回累乘器的值

#生成拉格朗日系数的函数——该函数用于生成拉格朗日系数矩阵U，用于多项式插值
def gen_Lagrange_coeffs(alpha_s, beta_s, p, is_K1=0):
    if is_K1 == 1:  #根据is_K1的值，确定alpha_s(1或者alpha_s的长度)
        num_alpha = 1
    else:
        num_alpha = len(alpha_s)
    #存储拉格朗日系数
    U = np.zeros((num_alpha, len(beta_s)), dtype='int64') #初始化U为一个全零矩阵，大小为num_alpha * len(beta_s)
    #         U = [[0 for col列 in range(len(beta_s))] for row行 in range(len(alpha_s))]
    # print(alpha_s)
    # print(beta_s)
    #定义w数组，用于存储每个beta_j与其他beta值的差的乘积
    #对于每个j，计算beta_s[j]与其他beta值的差的乘积，使用PI函数对p取模，存储在w[j]中
    w = np.zeros((len(beta_s)),dtype='int')
    for j in range(len(beta_s)):
        cur_beta = beta_s[j];
        den = PI([cur_beta - o for o in beta_s if cur_beta != o], p)
        w[j] = den
        
    l = np.zeros((num_alpha),dtype='int')
    for i in range(num_alpha):
        l[i] = PI([alpha_s[i] - o for o in beta_s], p)
        
    for j in range(len(beta_s)):
        for i in range(num_alpha):
            den = np.mod(np.mod(alpha_s[i]-beta_s[j],p)*w[j],p)
            U[i][j] = divmod(l[i],den,p)
            # for debugging
            # print(i,j,cur_beta,alpha_s[i])
            # print(test)
            # print(den,num) 
    return U.astype('int64') #获得拉格朗日系数矩阵U；大小为num_alpha * len(beta_s)

#作用——BGW协议编码，将输入数据X编码为N个多项式，每个多项式由T+1个点组成
#生成随机多项式系数R，在N个点上计算多项式值
def BGW_encoding(X, N, T, p):
    m = len(X)
    d = len(X[0])

    alpha_s = range(1, N + 1)
    alpha_s = np.int64(np.mod(alpha_s, p))
    X_BGW = np.zeros((N, m, d), dtype='int64')
    R = np.random.randint(p, size=(T + 1, m, d))
    R[0, :, :] = np.mod(X, p)

    for i in range(N):
        for t in range(T + 1):
            X_BGW[i, :, :] = np.mod(X_BGW[i, :, :] + R[t, :, :] * (alpha_s[i] ** t), p)
    return X_BGW

#生成BWG解码用的拉格朗日系数，对每个alpha计算特定的拉格朗日系数
def gen_BGW_lambda_s(alpha_s, p):
    lambda_s = np.zeros((1, len(alpha_s)), dtype='int64')

    for i in range(len(alpha_s)):
        cur_alpha = alpha_s[i];

        den = PI([cur_alpha - o for o in alpha_s if cur_alpha != o], p)
        num = PI([0 - o for o in alpha_s if cur_alpha != o], p)
        lambda_s[0][i] = divmod(num, den, p)
    return lambda_s.astype('int64')

#BWG协议解码函数，生成解码系数，使用系数对评估值进行线性组合
def BGW_decoding(f_eval, worker_idx, p):  # decode the output from T+1 evaluation points
    # f_eval     : [RT X d ]
    # worker_idx : [ 1 X RT]
    # output     : [ 1 X d ]

    # t0 = time.time()
    max = np.max(worker_idx) + 2
    alpha_s = range(1, max)
    alpha_s = np.int64(np.mod(alpha_s, p))
    alpha_s_eval = [alpha_s[i] for i in worker_idx]
    # t1 = time.time()
    # print(alpha_s_eval)
    lambda_s = gen_BGW_lambda_s(alpha_s_eval, p).astype('int64')
    # t2 = time.time()
    # print(lambda_s.shape)
    f_recon = np.mod(np.dot(lambda_s, f_eval), p)
    # t3 = time.time()
    # print 'time info for BGW_dec', t1-t0, t2-t1, t3-t2
    return f_recon


#LCC（线性一致性编码）编码函数，分割输入数据，添加随机块，使用拉格朗日插值编码
def LCC_encoding(X, N, K, T, p):
    m = len(X)
    d = len(X[0])
    # print(m,d,m//K)
    X_sub = np.zeros((K + T, m // K, d), dtype='int64')
    for i in range(K):
        X_sub[i] = X[i * m // K:(i + 1) * m // K:]
    for i in range(K, K + T):
        X_sub[i] = np.random.randint(p, size=(m // K, d))

    n_beta = K + T
    stt_b, stt_a = -int(np.floor(n_beta / 2)), -int(np.floor(N / 2))
    beta_s, alpha_s = range(stt_b, stt_b + n_beta), range(stt_a, stt_a + N)
    alpha_s = np.array(np.mod(alpha_s, p)).astype('int64')
    beta_s = np.array(np.mod(beta_s, p)).astype('int64')

    U = gen_Lagrange_coeffs(alpha_s, beta_s, p)
    # print U

    X_LCC = np.zeros((N, m // K, d), dtype='int64')
    for i in range(N):
        for j in range(K + T):
            X_LCC[i, :, :] = np.mod(X_LCC[i, :, :] + np.mod(U[i][j] * X_sub[j, :, :], p), p)
    return X_LCC

#带预定义随机数的LCC编码，使用外部提供的随机数R_而不是内部生成
def LCC_encoding_w_Random(X, R_, N, K, T, p):
    m = len(X)
    d = len(X[0])
    # print(m,d,m//K)
    X_sub = np.zeros((K + T, m // K, d), dtype='int64')
    for i in range(K):
        X_sub[i] = X[i * m // K:(i + 1) * m // K:]
    for i in range(K, K + T):
        X_sub[i] = R_[i - K, :, :].astype('int64')

    n_beta = K + T
    stt_b, stt_a = -int(np.floor(n_beta / 2)), -int(np.floor(N / 2))
    beta_s, alpha_s = range(stt_b, stt_b + n_beta), range(stt_a, stt_a + N)

    alpha_s = np.array(np.mod(alpha_s, p)).astype('int64')
    beta_s = np.array(np.mod(beta_s, p)).astype('int64')

    # alpha_s = np.int64(np.mod(alpha_s,p))
    # beta_s = np.int64(np.mod(beta_s,p))

    U = gen_Lagrange_coeffs(alpha_s, beta_s, p)
    # print U

    X_LCC = np.zeros((N, m // K, d), dtype='int64')
    for i in range(N):
        for j in range(K + T):
            X_LCC[i, :, :] = np.mod(X_LCC[i, :, :] + np.mod(U[i][j] * X_sub[j, :, :], p), p)
    return X_LCC

#部分节点的LCC编码，只对worker_idx指定的节点进行编码
def LCC_encoding_w_Random_partial(X, R_, N, K, T, p, worker_idx):
    m = len(X)
    d = len(X[0])
    # print(m,d,m//K)
    X_sub = np.zeros((K + T, m // K, d), dtype='int64')
    for i in range(K):
        X_sub[i] = X[i * m // K:(i + 1) * m // K:]
    for i in range(K, K + T):
        X_sub[i] = R_[i - K, :, :].astype('int64')

    n_beta = K + T
    stt_b, stt_a = -int(np.floor(n_beta / 2)), -int(np.floor(N / 2))
    beta_s, alpha_s = range(stt_b, stt_b + n_beta), range(stt_a, stt_a + N)
    alpha_s = np.array(np.mod(alpha_s, p)).astype('int64')
    beta_s = np.array(np.mod(beta_s, p)).astype('int64')
    alpha_s_eval = [alpha_s[i] for i in worker_idx]

    U = gen_Lagrange_coeffs(alpha_s_eval, beta_s, p)
    # print U

    N_out = U.shape[0]
    X_LCC = np.zeros((N_out, m // K, d), dtype='int64')
    for i in range(N_out):
        for j in range(K + T):
            X_LCC[i, :, :] = np.mod(X_LCC[i, :, :] + np.mod(U[i][j] * X_sub[j, :, :], p), p)
    return X_LCC

#LCC解码函数，生成解码矩阵，矩阵乘法恢复原始数据
def LCC_decoding(f_eval, f_deg, N, K, T, worker_idx, p):
    RT_LCC = f_deg * (K + T - 1) + 1

    n_beta = K  # +T
    stt_b, stt_a = -int(np.floor(n_beta / 2)), -int(np.floor(N / 2))
    beta_s, alpha_s = range(stt_b, stt_b + n_beta), range(stt_a, stt_a + N)
    alpha_s = np.array(np.mod(alpha_s, p)).astype('int64')
    beta_s = np.array(np.mod(beta_s, p)).astype('int64')
    alpha_s_eval = [alpha_s[i] for i in worker_idx]

    U_dec = gen_Lagrange_coeffs(beta_s, alpha_s_eval, p)

    # print U_dec 

    f_recon = np.mod((U_dec).dot(f_eval), p)

    return f_recon.astype('int64')

#生成加法秘密共享，生成n_out-1个随机数，最后一个数为负和取模，保证所有份额之和为原始值
def Gen_Additive_SS(d, n_out, p):
    # x_model should be one dimension

    temp = np.random.randint(0, p, size=(n_out - 1, d))
    # print temp

    last_row = np.reshape(np.mod(-np.sum(temp, axis=0), p), (1, d))
    Additive_SS = np.concatenate((temp, last_row), axis=0)
    # print np.mod(np.sum(Additive_SS,axis=0),p)

    return Additive_SS

# Add the function definition line and indent the following block
def LCC_encoding_with_points(X, alpha_s, beta_s, p):
    #用给定的点对数据进行编码；alpha_s——编码时使用的点；beta_s——目标点(编码后的数据在这些点上)；p——模数；确保计算在有限域内进行
    #编码的过程——相当于矩阵U(len(beta_s)*len(alpha_s))与输入矩阵X(m*d)相乘得到输出X_LCC(len(beta_s)*d)并取模p;
    #这里的m=len(alpha_s)
    m, d = np.shape(X)  #获取X的形状——m*d的矩阵

    # print alpha_s
    # print beta_s

    # for debugging LCC Enc & Dec
    # beta_s = np.concatenate((alpha_s, beta_s))
    # print beta_s
    #生成拉格朗日矩阵
    U = gen_Lagrange_coeffs(beta_s, alpha_s, p).astype('int')
    # print U；U的大小为len(beta_s)*len(alpha_s)
    #存储编码后的数据，大小为len(beta_s)*d
    X_LCC = np.zeros((len(beta_s), d), dtype='int')
    #遍历每个目标点beta_s[i],计算对应的编码数据

    #np.reshape(U[i, :], (1, len(alpha_s)))——将U的第i行转换为二维数组，大小为1*len(alpha_s)
    #np.dot(np.reshape(U[i, :], (1, len(alpha_s))), X)——将U的第i行与X的每一列元素进行点积运算，得到编码后的数据X_LCC[i, :]
    for i in range(len(beta_s)):
        X_LCC[i, :] = np.dot(np.reshape(U[i, :], (1, len(alpha_s))), X)
    #最终得到的X_LCC大小为len(beta_s)*d
    # print X
    # print np.mod(X_LCC, p)

    return np.mod(X_LCC, p)

#使用相同的点集对编码后的数据进行解码——用相同的点集获取另一个拉格朗日矩阵，与编码后的数据进行点积运算，并取模
#f_eval——编码后的数据；eval_points——解码时使用的点集；target_points——目标点集，beta_s；p——模数
#使用指定点集进行LCC解码，区别——显式指定评估点和目标点
def LCC_decoding_with_points(f_eval, eval_points, target_points, p):
    alpha_s_eval = eval_points #与编码时使用的点集相同
    beta_s = target_points  #目标点集
    #U_dec——大小为len(beta_s)*len(alpha_s_eval)
    U_dec = gen_Lagrange_coeffs(beta_s, alpha_s_eval, p)

    # print U_dec 
    #解码后的数据
    f_recon = np.mod((U_dec).dot(f_eval), p)
    # print f_recon

    return f_recon

#公钥生成
def my_pk_gen(my_sk, p, g):
    # print 'my_pk_gen option: g=',g
    if g == 0:
        return my_sk
    else:
        return np.mod(g ** my_sk, p)

#密钥协商，g=0时，直接相乘；g=1时，指数运算
def my_key_agreement(my_sk, u_pk, p, g):
    if g == 0:
        return np.mod(my_sk * u_pk, p)
    else:
        return np.mod(u_pk ** my_sk, p)
