#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=3 python3 main_train.py -lr=0.001 -layer=1 -hdd=512 -dr=0.2 -dec=Mem2SeqChitChat -bsz=256 -ds=chitchat -task=chitchat -t=