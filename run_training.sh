#!/usr/bin/env bash  

# sh run_training.sh mnist "./data/MNIST" lr "hetero" 1 8 0.01  #hetero——数据分布方式：异构；lr——模型：逻辑回归

#sh run_training.sh mnist "./data/MNIST" lr "centralized" 1 8 0.01  #centralized——数据分布方式：同构；cnn——模型：逻辑回归
#7个命令行参数待输入
if [ "$4" != "centralized" ] && [ "$4" != "hetero" ]; then
    echo "Error: partition_method must be 'centralized' or 'hetero'"
    exit 1
fi

DATASET=$1   #数据集名称
DATA_DIR=$2  #数据集路径
MODEL=$3  #模型名称
DISTRIBUTION=$4   #数据分布方式：同构、异构
EPOCH=$5   #训练轮数
BATCH_SIZE=$6    #训练批次大小
LR=$7   #学习率

#调用 Python3 解释器运行 main_trainer.py 脚本，并传入相应的参数(是否需要将python3修改为配置的conda环境lightsecagg?)
python3 ./main_trainer.py \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --data_dir "$DATA_DIR" \
  --partition_method "$DISTRIBUTION"  \
  --epochs "$EPOCH" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR"

  #向 Python 脚本传递 --model 参数，并将值设置为 $MODEL,7个值皆如此
