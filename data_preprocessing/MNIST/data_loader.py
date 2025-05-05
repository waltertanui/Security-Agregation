import json  #读取和解析.json格式的数据文件
import os    #处理文件和目录路径

import numpy as np   #进行高效数据处理和数值计算
import torch   #用于深度学习框架，将数据转换为张量以供模型训练

#作用——读取训练和测试数据目录中的.json文件，解析数据，并返回用户ID、组ID、训练数据和测试数据（假设训练和测试数据中的用户是相同的）
def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of non-unique client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    clients = []  
    groups = []  #初始化两个空列表，用于存储用户ID和组ID
    train_data = {}
    test_data = {}  #初始化两个字典，用于存储训练数据和测试数据
    #获取训练数据目录中的所有文件，并筛选出以.json结尾的文件
    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    #遍历每个.json文件，解析数据并更新用户ID、组ID和训练数据
    for f in train_files:
        file_path = os.path.join(train_data_dir, f) #构造完整的文件路径
        with open(file_path, 'r') as inf:  #打开.json文件并读取数据
            cdata = json.load(inf)  #解析.json文件中的数据，将文件内容转换为Python字典
        clients.extend(cdata['users'])  #将用户ID添加到clients列表中
        if 'hierarchies' in cdata: #如果数据中包含组信息，则将组信息添加到groups列表中
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data']) #将当前文件中的训练数据更新到train_data字典中
    #获取测试数据目录中的所有文件，并筛选出以.json结尾的文件
    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data']) #将当前文件中的测试数据更新到test_data字典中

    clients = sorted(cdata['users'])

    return clients, groups, train_data, test_data

#作用——将数据集划分为多个小批量，适用于训练过程中的批量处理
#首先将数据集随机打乱，然后按照指定的小批量大小将数据集划分为多个小批量，每个小批量转换为Pytorch张量并返回
def batch_data(data, batch_size):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    batch_data = list()
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i + batch_size]
        batched_y = data_y[i:i + batch_size]
        batched_x = torch.from_numpy(np.asarray(batched_x)).float()
        batched_y = torch.from_numpy(np.asarray(batched_y)).long()
        batch_data.append((batched_x, batched_y))
    return batch_data

#作用——根据设备ID加载MNIST数据集的分区数据（特定设备的MNIST数据集的训练和测试数据）
#构造了训练和测试数据的路径，并调用另一个函数load_partition_data_mnist来加载数据
def load_partition_data_mnist_by_device_id(batch_size,
                                           device_id,
                                           train_path="MNIST_mobile",
                                           test_path="MNIST_mobile"):
    train_path += '/' + device_id + '/' + 'train'
    test_path += '/' + device_id + '/' + 'test'
    return load_partition_data_mnist(batch_size, train_path, test_path)


def load_partition_data_mnist(batch_size,
                              train_path="./data/MNIST/train",
                              test_path="./data/MNIST/test"):
    users, groups, train_data, test_data = read_data(train_path, test_path)

    if len(groups) == 0:
        groups = [None for _ in users]
    train_data_num = 0
    test_data_num = 0
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    train_data_global = list()
    test_data_global = list()
    client_idx = 0
    for u, g in zip(users, groups):
        user_train_data_num = len(train_data[u]['x'])
        user_test_data_num = len(test_data[u]['x'])
        train_data_num += user_train_data_num
        test_data_num += user_test_data_num
        train_data_local_num_dict[client_idx] = user_train_data_num

        # transform to batches
        train_batch = batch_data(train_data[u], batch_size)
        test_batch = batch_data(test_data[u], batch_size)

        # index using client index
        train_data_local_dict[client_idx] = train_batch
        test_data_local_dict[client_idx] = test_batch
        train_data_global += train_batch
        test_data_global += test_batch
        client_idx += 1
    client_num = client_idx
    class_num = 10

    return client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
           train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num
