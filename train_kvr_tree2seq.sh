#!/usr/bin/env bash

LOG_DIR=log/

python3 distributed_train.py -lr=0.001 \
                              -layer=3 \
                              -hdd=128 \
                              -dr=0.2 \
                              -dec=Tree2Seq \
                              -bsz=16 \
                              -ds=kvr \
                              -task=kvr \
                              -t= \
                              --gpu_ranks 0 1 2 3 \
                              --distributed \
                              --worker 2 \
                              --print_freq 5 \
                              --world_size 4 |& tee $LOG_DIR/tree_log.txt