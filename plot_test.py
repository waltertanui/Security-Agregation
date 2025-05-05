import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import argparse  #解析命令行参数，使得程序可以接受命令行输入
import logging  #记录日志，方便跟踪、调试
import os     #与操作系统进行交互，如读取环境变量、创建目录等
import random  #生成随机数和随机选择
import socket  #用于网络通信，提供底层网络接口，用来创建和管理网络连接，进行数据传输等
import sys  #提供对Python解释器相关的操作，如获取命令行参数、退出程序等
import time  #处理与时间相关的操作，如获取当前时间、时间间隔测量、时间格式化等

import numpy as np  #数据处理与分析
import setproctitle  #第三方库，用于修改python进程的名称(在多进程或分布式任务中，通过设置进程名称，可以更方便地管理和调试进程)
import torch  #深度学习框架，提供张量操作、自动微分、神经网络模块等功能，用于构建和训练深度学习模型

# add the FedML root directory to the python path
from torch import nn  #神经网络模块，提供各种神经网络层（全连接层、卷积层、激活函数等）、损失函数、优化器等

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../")))
#os.getcwd()——返回当前工作目录的路径；os.path.abspath()——将相对路径转换为绝对路径(返回上一级目录)，os.path.join()——将多个路径组合成一个路径
#作用：将当前工作目录的上一级目录添加到sys.path的最前面，使得Python在导入模块时，首先查找这个目录;
# 如：假设当前工作目录是 /home/user/project/src，那么这段代码会将 /home/user/project 添加到 sys.path 的最前面；这样，当导入模块时，Python会优先在 /home/user/project 目录下查找模块
from data_preprocessing.MNIST.data_loader import load_partition_data_mnist
from data_preprocessing.FederatedEMNIST.data_loader import load_partition_data_federated_emnist
from data_preprocessing.fed_cifar100.data_loader import load_partition_data_federated_cifar100
from data_preprocessing.Landmarks.data_loader import load_partition_data_landmarks
#从4个.data_loader.py文件中导入各自的函数，用于加载和划分4个数据集

from model.lr import LogisticRegression  #导入逻辑回归模型，实现二分类任务
from model.cnn import CNN_DropOut  #导入卷积神经网络模型，实现图像分类任务
from model.mobilenet_v3 import MobileNetV3  #导入MobileNetV3模型(轻量级深度学习模型，适用于移动设备和资源受限环境)，实现图像分类任务
from model.efficientnet import EfficientNet  #导入EfficientNet模型（高效且准确的卷积神经网络模型），实现图像分类任务
from model.resnet_gn import resnet18  #导入ResNet18模型（深度残差网络模型），实现图像分类任务


def add_args(parser):  #使用argparse模块来定义和解析命令行参数，使脚本更加灵活和易于配置
    """
    parser : argparse.ArgumentParser  #(parser(解析器)的类型为argparse.ArgumentParser)
    return a parser added with args required by fit  #返回一个添加了所需参数的解析器
    """
    pass # function body is omitted

#利用给定数据集名称，来加载对应数据集，并返回一个包含数据集相关信息的列表
#args——一个包含训练参数的对象，通常是通过argparse解析的命令行参数
def load_data(args, dataset_name):  
    pass # function body is omitted  #将所有返回的变量组合成一个列表dataset，供后续使用


#根据模型名称和输出维度，创建模型；output_dim——输出维度，通常为类别数(整数)
def create_model(args, model_name, output_dim):
    pass # function body is omitted  #将创建的模型实例返回给调用者


if __name__ == "__main__":

    # parse python script input parameters
    parser = argparse.ArgumentParser()  #创建一个参数解析器，解析命令行参数
    args = add_args(parser)  #向解析器添加参数，并返回解析后的参数对象args，如·数据集、模型名称、训练轮次
    logging.info(args)  #打印解析后的参数，方便调试

    # customize the process name #设置当前进程标题为"FedAvg-Secure Aggregation"
    str_process_name = "FedAvg-Secure Aggregation"
    setproctitle.setproctitle(str_process_name)

    # customize the log format  配置日志模块，设置日志级别为DEBUG(这意味着所有级别的日志都会被记录)，并定义日志格式
    # logging.basicConfig(level=logging.INFO,
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S') #设置日志的输出格式
    hostname = socket.gethostname()  #获取当前主机名称

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(0)  #设置python随机数种子为0，以确保数据集划分的可重复性
    np.random.seed(0)  #设置numpy随机数种子
    torch.manual_seed(0)  #设置PyTorch的CPU随机数种子
    torch.cuda.manual_seed_all(0) #设置PyTorch的GPU随机数种子，确保所有GPU的随机性一致

    device = torch.device("cpu")  #设置训练设备为cpu，若要设成gpu，则cpu改为cuda(但需确保pytorch已正确安装了GPU版本)

    # load data
    dataset = load_data(args, args.dataset)
    [train_data_num, test_data_num, train_data_global, test_data_global,
     train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    #根据模型名称和数据集类别数dataset[7]创建模型实例，返回model(lr/cnn/resnet...)
    model = create_model(args, model_name=args.model, output_dim=dataset[7])
    #定义函数count_parameters_in_MB，用于计算模型参数数量，以百万为单位；
    def count_parameters_in_MB(model):
        pass # function body is omitted
    #调用函数获取模型大小，并记录到日志中
    model_size = count_parameters_in_MB(model)
    logging.info("model_size = %d" % model_size)

    sample_num_max = 0
    #遍历所有客户端的本地训练数据，计算每个客户端的样本数量，并记录到日志当中
    for client_idx in train_data_local_dict.keys():
        sample_num = len(train_data_local_dict[client_idx])
        logging.info("client_idx = %d, sample_num = %d" % (client_idx, sample_num))
    if args.dataset == "gld23k": #如果数据集为gld23k，选择（客户端236的数据，样本数为100）样本数量最多的客户端作为全局训练数据
        train_data = train_data_local_dict[216]  # 100 samples
    else:  #否则，选择（客户端8的数据）作为全局训练数据
        train_data = train_data_local_dict[8]
    logging.info("sample number = %d" % (len(train_data) * args.batch_size))
    #计算并记录实际的样本数量为选定客户端样本数量*批量大小

    model.to(device)  #将模型实例移动到训练设备上（cpu或gpu）
    model.train()  #将模型设置为训练模式
    #定义损失函数和优化器
    criterion = nn.CrossEntropyLoss().to(device) #创建交叉熵损失函数，用于计算模型输出与真实标签之间的损失，并将其移动到设备上
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)  #创建随机梯度下降（SGD）优化器，学习率为args.lr,动量为0.9，用于更新模型参数
    epoch_loss = []  #初始化空列表，记录每轮训练的平均损失


    batch_losses = []  #初始化空列表，记录每个批次的损失值
    batch_times = []  #初始化空列表，记录每个批次的训练时间
    batch_iterations = []  #初始化空列表，记录每个批次的迭代次数
    iteration_count = 0

    epoch_losses = []  # 新增：记录每一轮的平均损失
    epoch_times = []  # 新增：记录每一轮的训练时间

    time_start_training = time.time()  #记录开始的时间，用于计算总训练时间
    for epoch in range(args.epochs):  #进入训练循环，迭代args.epochs轮
        batch_loss = []
        epoch_start_time = time.time()  # 记录每一轮开始的时间
        for batch_idx, (x, labels) in enumerate(train_data): #每轮训练中，遍历数据集的每个批次
            # logging.info(images.shape)
            x, labels = x.to(device), labels.to(device)  #将输入数据和标签移动到训练设备上
            optimizer.zero_grad()  #清空梯度
            log_probs = model(x)  #将输入数据x输入模型，得到输出log_probs
            loss = criterion(log_probs, labels)  #计算损失函数
            loss.backward()  #反向传播，计算梯度
            optimizer.step()  #更新模型参数
            batch_loss.append(loss.item())  #将当前批次的损失值添加到batch_loss列表中
            
            #将当前批次的损失值添加到batch_loss列表中
            batch_losses.append(loss.item())
            batch_times.append(time.time() - time_start_training)
            batch_iterations.append(iteration_count)
            iteration_count += 1

        # 计算每一轮的平均损失
        if len(batch_loss) > 0:
            epoch_avg_loss = sum(batch_loss) / len(batch_loss)
            epoch_losses.append(epoch_avg_loss)
            epoch_times.append(time.time() - time_start_training)
            logging.info('Epoch: {} \tLoss: {:.6f}'.format(epoch, epoch_avg_loss))

    time_cost_on_training = time.time() - time_start_training  #计算训练总时间
    logging.info("time cost on training = %s" % str(time_cost_on_training))  #记录训练总时间到日志中

    # 绘制训练轮次与损失值的关系图
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(args.epochs), epoch_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch vs Loss')

    # 绘制训练时间 vs 损失值的图
    plt.subplot(1, 2, 2)
    plt.plot(epoch_times, epoch_losses)
    plt.xlabel('Training Time (s)')
    plt.ylabel('Loss')
    plt.title('Training Time vs Loss')

    plt.tight_layout()
    plt.show()