#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python3 main_test.py -dec=Mem2Seq -path=save/mem2seq-KVR/HDD128BSZ16DR0.2L3lr0.001Mem2Seq11.93 -bsz=16 -ds=kvr