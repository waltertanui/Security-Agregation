import math

import torch.nn as nn  #构建神经网络
import torch.utils.model_zoo as model_zoo  #加载预训练的模型权重

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']  #该模块中公开的类和函数包括ResNet、resnet18等。这些都是ResNet的不同变体，对应不同数量的层

from model.group_normalization import GroupNorm2d  #导入自定义的GroupNorm2d模块，用于实现分组归一化，而非传统的BatchNorm2d（批量归一化）

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}  #预训练模型的下载地址（可用）


def conv3x3(in_planes, out_planes, stride=1):  #定义函数conv3*3，创建一个3x3卷积层
    """3x3 convolution with padding"""  #输入通道数in_planes、输出通道数out_planes，步长stride；kernel_size=3——卷积核的大小是3x3，padding=1表示填充一圈0，使得输出的特征图尺寸与输入相同。bias=False——不使用偏置项
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,  #返回一个nn.Conv2d对象
                     padding=1, bias=False)


def norm2d(planes, num_channels_per_group=32):  #定义函数norm2d，创建一个归一化层
    print("num_channels_per_group:{}".format(num_channels_per_group))  #打印分组归一化中每组通道数
    if num_channels_per_group > 0:  #如果分组归一化中每组通道数大于0，则使用分组归一化GroupNorm2d；否则返回PyTorch内置的批量归一化层nn.BatchNorm2d对象
        return GroupNorm2d(planes, num_channels_per_group, affine=True,  #返回一个GroupNorm2d对象，输入通道数planes，每组通道数num_channels_per_group，affine=True——使用可学习的缩放和平移参数
                           track_running_stats=False)  #track_running_stats=False——不跟踪训练过程中的均值和方差，而是使用预训练模型中的均值和方差
    else:  
        return nn.BatchNorm2d(planes)


class BasicBlock(nn.Module):  #定义BasicBlock类，继承自nn.Module；BasicBlock——ResNet中的基础块，用于构建较浅的ResNet模型（如ResNet18）
    expansion = 1  #BasicBlock的输出通道数是输入通道数的1倍

    def __init__(self, inplanes, planes, stride=1, downsample=None,  #初始化函数，输入通道数inplanes，输出通道数planes，步长stride，下采样层downsample，分组归一化中的每组通道数group_norm
                 group_norm=0):
        super(BasicBlock, self).__init__()  #调用父类nn.Module的初始化函数
        self.conv1 = conv3x3(inplanes, planes, stride)  
        self.bn1 = norm2d(planes, group_norm)  #创建一个3x3卷积层和一个归一化层
        self.relu = nn.ReLU(inplace=True)  #创建一个ReLU激活函数，inplace=True——直接在输入数据上进行操作，不创建新的数据
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm2d(planes, group_norm)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 group_norm=0):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm2d(planes, group_norm)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = norm2d(planes, group_norm)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm2d(planes * 4, group_norm)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, group_norm=0):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm2d(64, group_norm)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],
                                       group_norm=group_norm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       group_norm=group_norm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       group_norm=group_norm)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       group_norm=group_norm)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, GroupNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.modules():
            if isinstance(m, Bottleneck):
                m.bn3.weight.data.fill_(0)
            if isinstance(m, BasicBlock):
                m.bn2.weight.data.fill_(0)

    def _make_layer(self, block, planes, blocks, stride=1, group_norm=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm2d(planes * block.expansion, group_norm),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            group_norm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, group_norm=group_norm))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
