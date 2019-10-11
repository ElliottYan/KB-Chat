#!/usr/bin/env bash

LOG_DIR=log/

hdd=512
drop=0.2
layer=2

EXPERIMENT_NAME=Tree_Ensemble_v5_hdd_${hdd}_ly_${layer}_drop_${drop}_add_clip_bs_8_match_utils
#_with_tree
export CUDA_LAUNCH_BLOCKING=1
export DATA_ROOT=data/

#export CUDA_VISIBLE_DEVICES=1,3

CUDA_VISIBLE_DEVICES=3 stdbuf -o0 python3 distributed_train.py -lr=0.001 \
                              -layer=$layer \
                              -hdd=$hdd \
                              -dr=$drop \
                              -dec=GLMP \
                              -bsz=8 \
                              -ds=kvr \
                              -task=kvr \
                              -t= \
                              --gpu_ranks 0 \
                              --experiment=$EXPERIMENT_NAME \
                              --worker 2 \
                              --max-epoch 50 \
                              --print_freq 5 \
                              --accumulate_step 1 \
                              --share-memnet \
                              --use-global \
                              --use-global-loss \
                              --use-sketch \
                              --world_size 1 |& tee $LOG_DIR/${EXPERIMENT_NAME}.txt

#                              --debug \
#                              --distributed \
#                              --distributed \
#                              --use-global \
#                              --use-global-loss \
#                              --gpu_ranks 0 \
#                              --use-global \
#                              --use-global-loss \
#                              --no-kb-embed \
