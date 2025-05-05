import numpy as np

#根据用户总数N和当前用户的排名rank来确认该用户所在的组号；假设N=1200，则n_users_group=11，n_group=110(109组为11人，1组为1人)
def get_group_idxset(N,rank):  #.astype(int)——截断小数部分，只保留整数部分
    n_users_group = np.ceil(np.log2(N)).astype(int)  #对N取以2为底的对数(希望每组的用户数量为2的幂次，方便后续处理)，向上取整，得到组内用户数
    n_group = np.ceil(float(N) / float(n_users_group)).astype(int)  #N除以组内用户数，向上取整，得到组数(若不能整除，则最后一组人数较少)
    #user_group_array[0-1210]——存储所有用户的索引，每个索引对应一个用户
    user_group_array = np.array(range(n_users_group*n_group),dtype='int')
    user_group_array[user_group_array>=N] = -1  #以N=1200为例，前1200个为正常用户索引，剩余部分置为-1，表示无效用户
    user_group_array = np.reshape(user_group_array,(n_group,n_users_group))
    #将一维数组user_group_array重新排列成二维数组，行数为组数n_group可能的值，列数为组内用户数n_users_group可能的值
    my_group_idx = np.floor(rank / n_users_group).astype(int)
    #假设rank=79，当前用户所在组数为79/11=7.18，向下取整，得到当前用户所在组号为7
    return user_group_array[my_group_idx,:]  #返回当前用户所在组的所有用户索引

#根据当前用户的组索引和层次结构，计算出其子层的用户索引集合，并处理可能的边界情况(如子层索引超出总层数或属于最后一层)，从而有效组织和管理用户的层次化分组信息，支持层次化的通信和数据传输
#在获取当前用户组索引的基础上，计算当前用户所在组的父层和子层的用户索引集合，适用于需要层次化分组和通信的场景
#每层包含多个组，每组包含多个用户
def get_list(N, rank):
    #N = 14  ，则n_users_layer = 3，n_layer = 5
    n_users_layer = np.ceil(np.log(N)).astype(int)  #对N取以e为底的对数，向上取整，得到每层用户数
    n_layer = np.ceil(float(N) / float(n_users_layer)).astype(int)  #N除以每层用户数，向上取整，得到层数(若不能整除，则最后一层人数较少)
    #user_group_array[0-14]——存储所有用户的索引，每个索引对应一个用户
    user_group_array = np.array(range(n_users_layer*n_layer),dtype='int')
    user_group_array[user_group_array>=N] = -1 #以N=14为例，前14个为正常用户索引，剩余部分(最后1个)置为-1，表示无效用户
    user_group_array = np.reshape(user_group_array,(n_layer,n_users_layer))  #将一维数组user_group_array重新排列成二维数组，行数为层数n_layer可能的值，列数为每层用户数n_users_layer可能的值
    #（5，3）
    last_group_idx = n_layer - 1 #最后一层的索引4(0-4)
    n_last_group = N - n_users_layer * (n_layer-1)  #最后一层的人数2(前4层为3人，最后一层为2人)
    #假设rank=5，当前用户所在组数为5/3=1.67，向下取整，得到当前用户所在组号为1
    #计算当前用户所在组号、父层(当前用户的层索引)和子层(当前拥护在该层内的顺序)的用户索引集合
    my_group_idx = rank / n_users_layer #当前用户所在组号1
    my_layer_idx = np.floor(np.log2(my_group_idx + 1)).astype(int) #当前用户的层索引1
    my_layer_order = (my_group_idx + 1 - 2**my_layer_idx).astype(int) #当前用户在该层内的顺序0?

    # Parent Setting
    prev_layer_idx = my_layer_idx - 1  #当前用户的父层索引0
    prev_layer_order = my_layer_order/2  #当前用户在该层内的顺序0
    if prev_layer_idx < 0:  #若当前用户为第一层，则没有父层，prev_group_idx和send_to_list设为 None
        prev_group_idx = None
        send_to_list = None
    else:  #否则，计算父层的组索引和对应的用户列表
        prev_group_idx = (2**prev_layer_idx) - 1 + prev_layer_order #0
        send_to_list = user_group_array[prev_group_idx,:] #[0,1,2]

    # Children Setting  #设置子层索引和对应的用户列表
    next_layer_idx = my_layer_idx + 1  #当前用户的子层索引2
    next_layer_order1 = my_layer_order*2  #第1个子层的顺序0
    next_layer_order2 = my_layer_order*2 + 1  #第2个子层的顺序1
    #计算两个子层的组索引和对应的用户列表
    next_group_idx1 = (2**next_layer_idx) - 1 + next_layer_order1 #3
    if next_group_idx1 >= n_layer: #如果子层索引超出总层数，设为 None
        next_group_idx1 = None
        rx_from_list1 = None
        rx_len1 = 0  #接收的用户数量为0
    elif next_group_idx1 == last_group_idx: #否则，若为最后一层，从user_group_array中提取最后一层的用户索引，但只提取有效用户索引；n_last_group——最后一组的有效用户人数
        rx_from_list1 = user_group_array[next_group_idx1,:n_last_group]
        rx_len1 = len(rx_from_list1)  #该子层有效用户的数量
    else: #若为前N-1层，从user_group_array中提取该层的所有用户索引；rx_from_list1——该子层用户的数量
        rx_from_list1 = user_group_array[next_group_idx1,:]#[6,7,8]
        rx_len1 = len(rx_from_list1)
        
    next_group_idx2 = (2**next_layer_idx) - 1 + next_layer_order2 #4
    if next_group_idx2 >= n_layer:
        next_group_idx2 = None
        rx_from_list2 = None
        rx_len2 = 0
    elif next_group_idx2 == last_group_idx:
        rx_from_list2 = user_group_array[next_group_idx2,:n_last_group]
        rx_len2 = len(rx_from_list2)#[9,10]
    else:
        rx_from_list2 = user_group_array[next_group_idx2,:]
        rx_len2 = len(rx_from_list2)

    #print rank,my_group_idx, '(',my_layer_idx, ',',my_layer_order,'), parents=',prev_group_idx, send_to_list,', children=(',next_group_idx1,'.',next_group_idx2,')', rx_from_list1,rx_from_list2

    return my_group_idx, send_to_list, rx_len1,rx_len2,rx_from_list1, rx_from_list2
    #返回当前用户所在组号、发送到父层的列表、两个子层的接收长度和接收列表