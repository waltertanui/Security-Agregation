import torch  #pytorch核心库，用于深度学习模型的构建和训练
import torch.nn as nn  #pytorch神经网络库，用于构建神经网络模型，如卷积层、全连接层和激活函数

# 定义CNN模型，用于联邦学习的原始FedAvg算法
class CNN_OriginalFedAvg(torch.nn.Module): # 继承自torch.nn.Module，表示这是一个神经网络模型
    """The CNN model used in the original FedAvg paper:
    "Communication-Efficient Learning of Deep Networks from Decentralized Data"
    https://arxiv.org/abs/1602.05629.

    The number of parameters when `only_digits=True` is (1,663,370), which matches
    what is reported in the paper.
    When `only_digits=True`, the summary of returned model is

    #输入：(28,28,1)单通道灰度图——>第一次卷积：(28,28,32)——>第一次池化：(14,14,32)
    ——>第二次卷积：(14,14,64)——>第二次池化：(7,7,64)——>全连接层：3136——>全连接层：10
    Model:
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    reshape (Reshape)            (None, 28, 28, 1)         0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 28, 28, 32)        832
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 14, 14, 64)        51264
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 7, 7, 64)          0
    _________________________________________________________________
    flatten (Flatten)            (None, 3136)              0  
    _________________________________________________________________
    dense (Dense)  (全连接)        (None, 512)               1606144 (输入：3136；输出：512；参数量：3136*512+512(可能为标签))
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                5130 (输入：512；输出：10；参数量：512*10+10（0-9标签）)
    =================================================================
    Total params: 1,663,370
    Trainable params: 1,663,370
    Non-trainable params: 0

    Args:
      only_digits: If True, uses a final layer with 10 outputs, for use with the
        digits only MNIST dataset (http://yann.lecun.com/exdb/mnist/).
        If False, uses 62 outputs for Federated Extended MNIST (FEMNIST)
        EMNIST: Extending MNIST to handwritten letters: https://arxiv.org/abs/1702.05373.
    Returns:
      A `torch.nn.Module`.
    """
    #整个网络结构的设计目的——在分布式数据环境下高效地进行模型训练，适用于像MNIST这样的手写数字识别任务，或者更复杂的FEMNIST任务

    #定义父类的初始化方法，这是继承类的常规操作
    def __init__(self, only_digits=True):
        super(CNN_OriginalFedAvg, self).__init__() #调用父类torch.nn.Module的初始化方法
        self.only_digits = only_digits #存储only_digits参数，用于确定输出层的大小
        self.conv2d_1 = torch.nn.Conv2d(1, 32, kernel_size=5, padding=2) 
        #定义第一个卷积层，输入通道数为1，输出通道数为32，卷积核大小为5，填充为2(保持输入和输出尺寸一致)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        #定义最大池化层，池化核大小为2，步长为2
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=5, padding=2)
        #定义第二个卷积层，输入通道数为32，输出通道数为64，卷积核大小为5，填充为2
        self.flatten = nn.Flatten() #定义展平层，将多维特征图展平为一维向量
        self.linear_1 = nn.Linear(3136, 512)  #定义第一个全连接层，输入特征数为3136(7*7*64)，输出特征数为512
        self.linear_2 = nn.Linear(512, 10 if only_digits else 62) #定义第二个全连接层，输入特征数为512，输出特征数为10(若为only_digits)或62(若only_digits为False)
        self.relu = nn.ReLU() #定义ReLU激活函数
        #self.softmax = nn.Softmax(dim=1) #定义softmax层，通常在损失函数中使用，不需要显式添加
      #用only_digits是否为True来决定输出层的大小，为True时输出层有10个神经元，用于识别MNIST的0-9类数字；否则，输出层有62个神经元，用于识别FEMNIST的62类数字和大小写字母

    #定义数据通过网络的传播过程(前向传播方法)
    def forward(self, x): #输入数据x
        x = torch.unsqueeze(x, 1) #在第一个维度上增加一个维度，将数据从(N, 28, 28)转换为(N, 1, 28, 28) —— (N, 28, 28)对应(batch_size, height, width)，(N, 1, 28, 28)对应(batch_size, channel, height, width)
        x = self.conv2d_1(x) #通过第一个卷积层
        x = self.relu(x) #通过ReLU激活函数
        x = self.max_pooling(x) #通过最大池化层
        x = self.conv2d_2(x)  #通过第二个卷积层
        x = self.relu(x) #通过ReLU激活函数
        x = self.max_pooling(x) #通过最大池化层
        x = self.flatten(x) #通过展平层
        x = self.relu(self.linear_1(x)) #通过第一个全连接层和ReLU激活函数
        x = self.linear_2(x) #通过第二个全连接层
        #x = self.softmax(self.linear_2(x)) #通常在损失函数中使用
        return x #返回输出结果
    #模型结构与原始FedAvg论文中的描述一致，包括两个卷积层、两个最大池化层、两个全连接层和ReLU激活函数；总参数数量为1663370；通过only_digits参数可以灵活调整输出层大小，适应不同的数据集，如MNIST或FEMNIST。

#Dropout正则化技术——防止神经网络过拟合：通过在训练过程中随机丢弃部分神经元，减少模型对特定训练样本的依赖，提高模型的泛化能力
#Dropout基本思想：每次训练迭代中，随机选择部分神经元并在本次迭代中禁用它们，即它们的输出不传递到下一层；这样，网络在每一次训练时使用不同子网络，这些子网络共享权重参数
#Dropout通常应用于全连接层，也适用于卷积层和循环层（仅在训练时起作用，测试或推理时禁用Dropout，所有神经元正常工作）
#Dropout劣势——在每次训练迭代中，都需要进行随机的神经元丢弃操作，会增加一定的计算开销
class CNN_DropOut(torch.nn.Module): #用于图像分类的CNN模型，结合了Dropout正则化技术
    """
    Recommended model by "Adaptive Federated Optimization" (https://arxiv.org/pdf/2003.00295.pdf)
    Used for EMNIST experiments.
    When `only_digits=True`, the summary of returned model is
    ```
    Model:
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    reshape (Reshape)            (None, 28, 28, 1)         0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 26, 26, 32)        320 (卷积核3*3——26=28-3+1;320=32个卷积核*（卷积核的3*3个参数+1个偏置项）)
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 24, 24, 64)        18496 (再次卷积，得24；)
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 12, 12, 64)        0 (减半)
    _________________________________________________________________
    dropout (Dropout)            (None, 12, 12, 64)        0
    _________________________________________________________________
    flatten (Flatten)            (None, 9216)    (一维化)   0
    _________________________________________________________________
    dense (Dense)   (全连接)      (None, 128)               1179776(=9216*128+128)
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 128)               0
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                1290(=128*10+10)
    =================================================================
    Total params: 1,199,882
    Trainable params: 1,199,882
    Non-trainable params: 0
    ```
    Args:
      only_digits: If True, uses a final layer with 10 outputs, for use with the
        digits only MNIST dataset (http://yann.lecun.com/exdb/mnist/).
        If False, uses 62 outputs for Federated Extended MNIST (FEMNIST)
        EMNIST: Extending MNIST to handwritten letters: https://arxiv.org/abs/1702.05373.
    Returns:
      A `torch.nn.Module`.
    """
    #用only_digits是否为True来决定输出层的大小，为True时输出层有10个神经元，用于识别MNIST的0-9类数字；否则，输出层有62个神经元，用于识别FEMNIST的62类数字和大小写字母
    #__init__定义模型的各个层;forward定义数据通过这些层的顺序
    def __init__(self, only_digits=True):
        super(CNN_DropOut, self).__init__()
        self.conv2d_1 = torch.nn.Conv2d(1, 32, kernel_size=3) #输入通道为1，输出通道为32
        self.max_pooling = nn.MaxPool2d(2, stride=2)  #用于降低特征图的尺寸，减少计算量
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=3) #将通道数由32增加到64，继续提取更复杂的特征
        self.dropout_1 = nn.Dropout(0.25)  #定义第一个Dropout层，丢弃率为0.25，用于防止过拟合
        self.flatten = nn.Flatten() #一维化展平，为全连接层做准备
        self.linear_1 = nn.Linear(9216, 128) #定义第一个全连接层，输入特征数为9216，输出特征数为128（将特征映射到潜在的类别空间）
        self.dropout_2 = nn.Dropout(0.5) #定义第二个Dropout层，丢弃率为0.5，用于防止过拟合
        self.linear_2 = nn.Linear(128, 10 if only_digits else 62) #定义第二个全连接层（输出层），输入特征数为128，输出特征数为10或62，取决于only_digits参数的值
        self.relu = nn.ReLU() #定义ReLU激活函数，用于引入非线性，使模型能够学习更复杂的特征
        #self.softmax = nn.Softmax(dim=1)

    #forward定义数据通过这些层的顺序
    def forward(self, x):
        x = torch.unsqueeze(x, 1) #增加一个维度，使输入数据x变成四维张量变成四维张量（批次大小，通道数，高度，宽度）
        x = self.conv2d_1(x) #依次通过两个卷积层和激活函数，提取特征
        x = self.relu(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x) #最大池化，进一步降低特征图的尺寸
        x = self.dropout_1(x) #随机丢弃部分神经元，防止过拟合
        x = self.flatten(x) #将特征图展平成一维向量
        x = self.linear_1(x) #通过全连接层，将特征映射到潜在类别空间
        x = self.relu(x) #激活函数
        x = self.dropout_2(x) #再次随机丢弃部分神经元，防止过拟合
        x = self.linear_2(x) #通过输出层，得到最终的分类结果
        #x = self.softmax(self.linear_2(x))
        return x #返回类别预测结果
