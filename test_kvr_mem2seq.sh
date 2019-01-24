#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2 python3 main_test.py -dec=Mem2Seq -path=save/mem2seq-KVR/HDD256BSZ16DR0.2L3lr0.001Mem2Seq12.04 -bsz=16 -ds=kvr