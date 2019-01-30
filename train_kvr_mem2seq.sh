#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python3 main_train.py -lr=0.001 -layer=1 -hdd=256 -dr=0.2 -dec=Mem2Seq -bsz=16 -ds=kvr -task=kvr -t=