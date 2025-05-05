# Environment Setup
```
conda create -n lightsecagg  创建一个虚拟环境
conda activate lightsecagg
conda install numpy
conda install mpi4py    安装失败，改用pip安装成功（mpi4py--用于并行计算）
pip install torch        （torch--用于深度学习）
pip install torchvision    （torchvision--用于处理图像数据）
pip install -r requirements.txt
```

# Experimental Design
We assume the edge device in FL is the following configuration and train with CPU device.

Hardware: MacBook Pro (16-inch, 2019), 2.3 GHz 8-Core Intel Core i9

CPU training: torch.device("cpu")   （将pytorch的计算和运行设备设置为CPU，而非GPU：torch.device(cuda) ）

(Note: please download data at the "./data" folder before training.)

1. Linear model on small data: MNIST + Logistic Regression    （在小型数据集MNIST上进行逻辑回归训练）

```
sh run_training.sh mnist "./data/MNIST" lr "hetero" 1 2 0.01

INFO:root:LogisticRegression + MNIST
INFO:root:model_size = 7850
INFO:root:sample number = 72
INFO:root:time cost on training = 0.0922391414642334
```

2. Small CNN model on small data: Federated MNIST + CNN (suggested by the original FedAvg paper)    （在小型数据集Federated MNIST上训练CNN模型）
```
sh run_training.sh femnist "./data/FederatedEMNIST/datasets" cnn "hetero" 1 2 0.01


INFO:root:load_data. dataset_name = femnist
INFO:root:class_num = 62
INFO:root:create_model. model_name = cnn, output_dim = 62
INFO:root:CNN + FederatedEMNIST
INFO:root:model_size = 1206590
INFO:root:sample number = 382
INFO:root:time cost on training = 2.276911735534668 seconds

```

3. Efficient edge model (MobileNetV3) + Fed-CIFAR100 low resolution of images (24x24)     （使用MobileNetV3模型在CIFAR100数据集上训练，低图像分辨率24x24）

```
sh run_training.sh fed_cifar100 "./data/fed_cifar100/datasets" mobilenet_v3 "hetero" 1 2 0.01

INFO:root:create_model. model_name = mobilenet_v3, output_dim = 500
INFO:root:model_size = 3111462
INFO:root:sample number = 200
INFO:root:time cost on training = 26.284486770629883 seconds

```
    
4. Efficient Edge Model (EfficientNet-B0) + GLD23K (high resolution image: 224 x 224)     （使用EfficientNet-B0模型在GLD23K数据集上训练，高图像分辨率224x224）

```
# data is located at ./data/gld/data_user_dict

sh run_training.sh gld23k "./data/gld/" efficientnet "hetero" 1 2 0.01

INFO:root:load_data. dataset_name = gld23k
INFO:root:create_model. model_name = efficientnet, output_dim = 203
INFO:root:model_size = 5288548
INFO:root:sample number = 200
INFO:root:time cost on training = 232.3275842666626

```

# Run Experiments
```
sh run_lightsecagg.sh     （运行训练脚本，包括四个数据集的训练：MNIST、Federated MNIST、Fed-CIFAR100、gld23k）

For each MPI_xxx.py, we can set the number of clients(=N), number of parameters(=d), and communication bandwidht (Mbps) by using arguments, i.e.,      (设置客户端数量N，参数数量d，通信带宽Mbps)

mpirun -np (N+1) python MPI_LightSecAgg.py N d comm_bandwidth    （举例说明）

Ex) ResNet18 (#params=11.4M), Bandwidth = 100Mbps, N={8,16,25,50,100,150,200}
mpirun -np 9 python MPI_LightSecAgg.py 8 11511784 100
mpirun -np 17 python MPI_LightSecAgg.py 16 11511784 100
mpirun -np 26 python MPI_LightSecAgg.py 25 11511784 100
mpirun -np 51 python MPI_LightSecAgg.py 50 11511784 100
mpirun -np 101 python MPI_LightSecAgg.py 100 11511784 100
mpirun -np 151 python MPI_LightSecAgg.py 150 11511784 100
mpirun -np 201 python MPI_LightSecAgg.py 200 11511784 100
```


mpirun -np 9 python MPI_LightSecAgg.py 8 10000000 100
