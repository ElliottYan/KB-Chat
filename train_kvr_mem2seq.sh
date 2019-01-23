#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python3 main_train.py -lr=0.001 -layer=3 -hdd=512 -dr=0.3 -dec=Mem2Seq -bsz=16 -ds=kvr -task=kvr -t=