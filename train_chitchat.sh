#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=3 python3 main_train.py -lr=0.001 -layer=1 -hdd=128 -dr=0.2 -dec=Mem2Seq -bsz=16 -ds=chitchat -task=chitchat -t=