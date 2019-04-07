#!/usr/bin/env bash

LOG_DIR=log/

CUDA_VISIBLE_DEVICES=2,3 python3 distributed_train.py -lr=0.001 \
                                                      -layer=3 \
                                                      -hdd=128 \
                                                      -dr=0.2 \
                                                      -dec=Mem2Seq \
                                                      -bsz=8 \
                                                      -ds=kvr \
                                                      -task=kvr \
                                                      -t= \
                                                      --gpu_ranks 0 1 \
                                                      --worker 2 \
                                                      --print_freq 5 \
                                                      --world_size 1 |& tee $LOG_DIR/mem_log.txt
#                                                      --distributed \
