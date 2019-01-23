#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2 python3 main_test.py -dec=Mem2Seq -path=save/mem2seq-KVR/HDD128BSZ8DR0.2L1lr0.001Mem2Seq0.15 -bsz=16 -ds=kvr