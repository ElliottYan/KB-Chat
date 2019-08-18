#!/usr/bin/env bash

LOG_DIR=log/

hdd=512
drop=0.2
layer=2

EXPERIMENT_NAME=Tree_Ensemble_v5_hdd_${hdd}_ly_${layer}_drop_${drop}_add_global_loss

python3 distributed_train.py -lr=0.001 \
                              -layer=$layer \
                              -hdd=$hdd \
                              -dr=$drop \
                              -dec=Tree2Seq \
                              -bsz=64 \
                              -ds=kvr \
                              -task=kvr \
                              -t= \
                              --experiment=$EXPERIMENT_NAME \
                              --gpu_ranks 0 1 2 3 \
                              --worker 2 \
                              --max-epoch 50 \
                              --distributed \
                              --print_freq 5 \
                              --debug \
                              --use-global-loss \
                              --world_size 4 |& tee $LOG_DIR/${EXPERIMENT_NAME}.txt

#                              --no-kb-embed \
