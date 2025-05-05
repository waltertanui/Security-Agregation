#!/bin/bash
CLIENT_NUM=100  #定义客户端数量，并将其值设为100

PROCESS_NUM="$( $CLIENT_NUM + 1)"  #expr显示过时，删去expr  #9个进程：8个客户端+1个服务器

mpirun -np "$PROCESS_NUM" python new.py $CLIENT_NUM  #运行MPI_LightSecAgg.py脚本，并传入客户端数量作为参数PROCESS_NUM=`expr $CLIENT_NUM + 1`    #定义进程数量，并将其值设为客户端数量加1——101个进程：100个客户端+1个服务器
