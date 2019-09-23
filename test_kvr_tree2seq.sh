#!/usr/bin/env bash

LOG_DIR=log/

hdd=512
drop=0.2
layer=2

EXPERIMENT_NAME=Tree_Ensemble_v5_hdd_${hdd}_ly_${layer}_drop_${drop}_change_data_source

RESUME_PATH=./model/${EXPERIMENT_NAME}_best.pt
#RESUME_PATH=./model/Tree_Ensemble_v5_best.pt
export DATA_ROOT=data/
export CUDA_LAUNCH_BLOCKING=1

python3 distributed_train.py -lr=0.001 \
                              -layer=$layer \
                              -hdd=$hdd \
                              -dr=$drop \
                              -dec=Tree2Seq \
                              -bsz=64 \
                              -ds=kvr \
                              -task=kvr \
                              -t= \
                              --mode=test \
                              --experiment=$EXPERIMENT_NAME \
                              --gpu_ranks 0 1 2 3 \
                              --worker 2 \
                              --resume=$RESUME_PATH \
                              --max-epoch 50 \
                              --print_freq 5 \
                              --world_size 1 |& tee $LOG_DIR/${EXPERIMENT_NAME}_test.txt

#                              --no-kb-embed \
#                              --distributed \
#                              -add-norm \
#                              --debug \

