#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2 python3 main_train.py -lr=0.001 -layer=3 -hdd=128 -dr=0.2 -dec=Tree2Seq -bsz=16 -ds=kvr -task=kvr -t=