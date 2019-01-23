#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python3 main_test.py -dec=Mem2Seq -path=save/mem2seq-KVR/HDD256BSZ16DR0.3L3lr0.001Mem2Seq0.13 -bsz=16 -ds=kvr