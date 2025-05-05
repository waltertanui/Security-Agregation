import logging  #记录日志信息，方便调试
import os  #处理文件路径操作

import h5py  #读取h5文件
import numpy as np  #处理数值数据，尤其是多维数组
import torch  #PyTorch深度学习框架
import torch.utils.data as data  #PyTorch数据加载工具

logging.basicConfig()  #设置日志的基本配置，获取默认的根日志记录器，设置日志级别为INFO，可记录INFO级别及以上的日志信息（包括：INFO、WARNING、ERROR、CRITICAL）
logger = logging.getLogger()
logger.setLevel(logging.INFO)

#存储训练和测试的客户端id，初始值为None，数量都为3400;
#默认批量大小为20
#训练和测试数据的.h5文件分别为：fed_emnist_train.h5和fed_emnist_test.h5
client_ids_train = None
client_ids_test = None
DEFAULT_TRAIN_CLIENTS_NUM = 3400
DEFAULT_TEST_CLIENTS_NUM = 3400
DEFAULT_BATCH_SIZE = 20
DEFAULT_TRAIN_FILE = 'fed_emnist_train.h5'
DEFAULT_TEST_FILE = 'fed_emnist_test.h5'

# group name defined by tff in h5 file（定义.h5文件中的组名）
_EXAMPLE = 'examples' #训练数据集和测试数据集的组名
_IMGAE = 'pixels'
_LABEL = 'label'
#分别表示：包含客户端数据的组名、客户端数据中的图像数据、客户端数据中的标签数据

#从h5文件中读取数据，并创建pytorch的数据加载器DataLoader
#def get_dataloader(数据集名称、数据集路径、训练集批量大小、测试集批量大小、要加载的客户端id——若为None，则加载所有客户端的数据):
def get_dataloader(dataset, data_dir, train_bs, test_bs, client_idx=None):
#以只读模式打开训练和测试的.h5文件，得到文件路径./datasets/xxx.h5
    train_h5 = h5py.File(os.path.join(data_dir, DEFAULT_TRAIN_FILE), 'r')
    test_h5 = h5py.File(os.path.join(data_dir, DEFAULT_TEST_FILE), 'r')
    #初始化图像数据_x与标签数据_y列表
    train_x = []
    test_x = []
    train_y = []
    test_y = []
#若客户端id为None，加载所有客户端数据，训练客户端id设置为整体client_ids_train(line 16);
#否则，只加载指定客户端id的数据，训练客户端id设置为[client_ids_train[client_idx]]
    # load data
    if client_idx is None:
        # get ids of all clients
        train_ids = client_ids_train
        test_ids = client_ids_test
    else:
        # get ids of single client
        train_ids = [client_ids_train[client_idx]]
        test_ids = [client_ids_test[client_idx]]

#np.vstack：将每个客户端的数据垂直堆叠成一个大的NumPy数组
    # load data in numpy format from h5 file
    #train_h5——打开的train.h5文件对象，指向训练数据集文件fed_emnist_train.h5；
    #train_h5文件中，_EXAMPLE组下包含多个子组，每个子组对应一个客户端；每个客户端子组下有两个数据集：_IMGAE图像数据和_LABEL标签数据；
    train_x = np.vstack([train_h5[_EXAMPLE][client_id][_IMGAE][()] for client_id in train_ids])
    train_y = np.vstack([train_h5[_EXAMPLE][client_id][_LABEL][()] for client_id in train_ids]).squeeze()
    test_x = np.vstack([test_h5[_EXAMPLE][client_id][_IMGAE][()] for client_id in test_ids])
    test_y = np.vstack([test_h5[_EXAMPLE][client_id][_LABEL][()] for client_id in test_ids]).squeeze()
    #.sqeeze()：去掉标签数据中的冗余维度，确保标签数据的形状适合模型输入
#得到训练集和测试集的图像数据（numpy数组）train_x、test_x和标签数据train_y、test_y

    # dataloader
    #将训练集和测试集的图像数据和标签数据转换为PyTorch的Tensor，并创建数据加载器DataLoader
    #torch.tensor：将numpy数组转换为PyTorch的Tensor
    train_ds = data.TensorDataset(torch.tensor(train_x), torch.tensor(train_y, dtype=torch.long))
    train_dl = data.DataLoader(dataset=train_ds,  #tensor
                               batch_size=train_bs,  #设置批量大小
                               shuffle=True,  #是否打乱数据顺序：是
                               drop_last=False)  #是否丢弃最后一个不完整的批次：否

    test_ds = data.TensorDataset(torch.tensor(test_x), torch.tensor(test_y, dtype=torch.long))
    test_dl = data.DataLoader(dataset=test_ds,
                                  batch_size=test_bs,
                                  shuffle=True,
                                  drop_last=False)

    train_h5.close()
    test_h5.close()
    return train_dl, test_dl
#关闭.h5文件并返回数据加载器train_dl和test_dl(包含tensor、批量大小、打乱数据顺序、不丢弃最后一个不完整批次的设置)

#用于加载分布式联邦学习EMNIST数据集的函数(用当前进程id是否为0，来区分全局进程和本地进程)
def load_partition_data_distributed_federated_emnist(process_id, dataset, data_dir, 
                                                     batch_size=DEFAULT_BATCH_SIZE):
#process_id：表示当前进程id，用于区分全局进程和本地进程；batch_size默认值为20(DEFAULT_BATCH_SIZE=20)(line 20)
    if process_id == 0:  #若当前进程id为0，进行全局数据集加载(全局进程)
        # get global dataset  #get_dataloader——line 32，用(当前进程id-1)作为客户端id 
        train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size, process_id - 1)  
        #获取全局训练和测试数据加载器train_data_global和test_data_global
        train_data_num = len(train_data_global) #计算全局训练数据集的大小
        # logging.info("train_dl_global number = " + str(train_data_num))  #打印全局训练和测试数据集的大小
        # logging.info("test_dl_global number = " + str(test_data_num))
        train_data_local = None  #设置本地训练和测试数据集，None表示全局进程中没有本地数据集
        test_data_local = None
        local_data_num = 0  #本地数据集大小设置为0
    else:
        # get local dataset
        train_file_path = os.path.join(data_dir, DEFAULT_TRAIN_FILE) #line 21；DEFAULT_TRAIN_FILE = 'fed_emnist_train.h5'
        test_file_path = os.path.join(data_dir, DEFAULT_TEST_FILE)  #line 22；DEFAULT_TEST_FILE = 'fed_emnist_test.h5'
        with h5py.File(train_file_path, 'r') as train_h5, h5py.File(test_file_path, 'r') as test_h5:
            #以只读模式打开训练和测试的.h5文件，得到文件路径./datasets/xxx.h5
            global client_ids_train, client_ids_test  #client_ids_train、client_ids_test——训练和测试数据集中所有客户端的id列表
            #global——声明client_ids_train、client_ids_test为全局变量，故在函数外部也可以访问到这些变量
            client_ids_train = list(train_h5[_EXAMPLE].keys()) #1个EXAMPLE对应1组客户端，取其值为所有客户端id列表
            client_ids_test = list(test_h5[_EXAMPLE].keys())
        #赋值本地训练和测试数据集，其大小皆设置为本地训练数据集的大小；全局训练和测试数据集设置为None，表示本地进程中没有全局数据集
        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size, process_id - 1)
        train_data_num = local_data_num = len(train_data_local)
        train_data_global = None
        test_data_global = None

    # class number
    train_file_path = os.path.join(data_dir, DEFAULT_TRAIN_FILE) #同 line 100
    #再次以只读模式打开训练集的.h5文件，计算训练集的类别数量，并使用logging.info()打印类别数量
    with h5py.File(train_file_path, 'r') as train_h5:
        #line 18：DEFAULT_TRAIN_CLIENTS_NUM=3400，遍历
        class_num = len(np.unique([train_h5[_EXAMPLE][client_ids_train[idx]][_LABEL][0] for idx in range(DEFAULT_TRAIN_CLIENTS_NUM)]))  
        logging.info("class_num = %d" % class_num)

    return DEFAULT_TRAIN_CLIENTS_NUM, train_data_num, train_data_global, test_data_global, local_data_num, train_data_local, test_data_local, class_num
    #返回训练客户端数量、训练数据集大小(全局或本地取决于process_id是0还是非0)、全局训练和测试数据集、本地数据集大小、本地训练和测试数据集、类别数量

#用于加载联邦学习EMNIST数据集，并进行数据的分割和处理的函数
#加载并分割EMNIST数据集，为每个客户端分配本地数据集，同时创建全局数据集和数据加载器；此外，计算数据集中的类别数量，以便后续模型训练和评估使用
def load_partition_data_federated_emnist(dataset, data_dir, batch_size):
    
    # client ids
    train_file_path = os.path.join(data_dir, DEFAULT_TRAIN_FILE)
    test_file_path = os.path.join(data_dir, DEFAULT_TEST_FILE)
    with h5py.File(train_file_path, 'r') as train_h5, h5py.File(test_file_path, 'r') as test_h5:
        global client_ids_train, client_ids_test
        client_ids_train = list(train_h5[_EXAMPLE].keys())
        client_ids_test = list(test_h5[_EXAMPLE].keys())
    #获取训练和测试数据集中所有客户端的id列表
    
    # local dataset
    data_local_num_dict = dict() #每个用户的本地数据数量
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    #初始化字典，用于存储训练和测试数据集中每个客户端的本地数据集大小、训练和测试数据加载器
    for client_idx in range(DEFAULT_TRAIN_CLIENTS_NUM):
        #遍历3400个训练客户端，获取训练和测试数据加载器，并计算本地数据集大小=训练数据集大小+测试数据集大小
        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size, client_idx)
        local_data_num = len(train_data_local) + len(test_data_local)
        data_local_num_dict[client_idx] = local_data_num #存储每个用户的本地数据数量
        # logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))
        # logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
        #     client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local #存储每个用户的训练和测试数据加载器
        test_data_local_dict[client_idx] = test_data_local

    # global dataset
    #创建全局训练(由所有客户端的本地训练数据集组成)和测试(由所有客户端的本地测试数据集组成)数据集，并计算全局训练和测试数据集的大小
    train_data_global = data.DataLoader(
                data.ConcatDataset(
                    list(dl.dataset for dl in list(train_data_local_dict.values()))
                ),
                batch_size=batch_size, shuffle=True) #打乱数据集
    train_data_num = len(train_data_global.dataset)
    
    test_data_global = data.DataLoader(
                data.ConcatDataset(
                    list(dl.dataset for dl in list(test_data_local_dict.values()) if dl is not None)
                ),
                batch_size=batch_size, shuffle=True)
    test_data_num = len(test_data_global.dataset)
    
    # class number
    train_file_path = os.path.join(data_dir, DEFAULT_TRAIN_FILE)
    with h5py.File(train_file_path, 'r') as train_h5:
        class_num = len(np.unique([train_h5[_EXAMPLE][client_ids_train[idx]][_LABEL][0] for idx in range(DEFAULT_TRAIN_CLIENTS_NUM)]))
        # logging.info("class_num = %d" % class_num)
        #遍历每个客户端的第一个样本标签，计算类别数量
    return DEFAULT_TRAIN_CLIENTS_NUM, train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num
        #返回训练客户端数量、全局训练和测试数据集大小、全局训练和测试数据集、每个用户的本地数据集大小、每个用户的训练和测试数据加载器、类别数量
def load_partition_data_mnist(batch_size,partition_method="hetero"):
    # 加载数据集，支持集中式和异构式两种分区方式
    #batch_size：批量大小
    #return ：数据集相关信息
    from torchvision import datasets,transforms

    #数据预处理
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    #加载完整数据集
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    if partition_method == "centralized":
        # 集中式训练 - 所有数据在一个客户端上
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        # 返回数据结构与异构式一致，但只有一个客户端
        train_data_num = len(train_dataset)
        test_data_num = len(test_dataset)
        train_data_global = train_loader
        test_data_global = test_loader
        train_data_local_num_dict = {0: train_data_num}
        train_data_local_dict = {0: train_loader}
        test_data_local_dict = {0: test_loader}
        class_num = 10
        
        return 1, train_data_num, test_data_num, train_data_global, test_data_global, \
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num
    
    else:
        # 异构式训练 - 每个客户端拥有部分数据
        # 每个客户端拥有10%的数据
        num_clients = 10
        client_train_datasets=[]
        client_test_datasets=[]

        #随机分配训练数据
        indices=torch.randperm(len(train_dataset))
        splits=torch.tensor_split(indices,num_clients)

        for i in range(num_clients):
            subset=data.Subset(train_dataset,splits[i])
            client_train_datasets.append(subset)

        #随机分配测试数据
        indices=torch.randperm(len(test_dataset))
        splits=torch.tensor_split(indices,num_clients)

        for i in range(num_clients):
            subset=data.Subset(test_dataset,splits[i])
            client_test_datasets.append(subset)

        #创建数据加载器
        train_data_local_dict = {}
        test_data_local_dict = {}
        train_data_local_num_dict = {}

        for i in range(num_clients):
            train_loader=data.DataLoader(client_train_datasets[i],batch_size=batch_size,shuffle=True)
            test_loader=data.DataLoader(client_test_datasets[i],batch_size=batch_size,shuffle=False)
            
            train_data_local_dict[i] = train_loader
            test_data_local_dict[i] = test_loader
            train_data_local_num_dict[i] = len(client_train_datasets[i])

        # global dataset
        train_data_global = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_data_global = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        train_data_num = len(train_dataset)
        test_data_num = len(test_dataset)
        class_num = 10

        return num_clients, train_data_num, test_data_num, train_data_global, test_data_global, \
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num

