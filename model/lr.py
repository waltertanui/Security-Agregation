import torch #pytorch——流行的深度学习框架，支持神经网络的构建与训练


class LogisticRegression(torch.nn.Module):  #定义了一个名为LogisticRegression的类，它继承自torch.nn.Module；Module——PyTorch中构建神经网络的基本模块，每个自定义的神经网络模型都应该继承自Module
    def __init__(self, input_dim, output_dim):  #初始化模型参数
        super(LogisticRegression, self).__init__()   #调用父类的初始化方法，确保父类的初始化工作正确执行
        self.linear = torch.nn.Linear(input_dim, output_dim)  #定义一个线性层（全连接层），输入维度为input_dim，输出维度为output_dim
                                                              #线性层作用——将输入特征通过线性变换（矩阵乘法或加法）得到一个线性组合
    def forward(self, x):                         #定义前向传播过程，即输入数据x经过模型后如何得到输出
        outputs = torch.sigmoid(self.linear(x))   #输入`x`通过线性层得到一个线性组合self.linear(x)，然后通过sigmoid函数得到输出outputs，outputs的值在0-1之间，表示每个样本属于正常的概率
        return outputs        #每个样本属于各个类别的概率
    
# in all: 定义一个逻辑回归模型，它由一个线性层和一个sigmoid激活函数组成。在MNIST数据集中，输入是28x28的图像（784个像素），输出是10个类别（0-9）。因此，`input_dim`设置为784，`output_dim`设置为10。
#通过在MNIST数据集上进行训练，模型可以学习到如何将输入图像映射到对应的数字类别，从而实现手写数字的识别。
