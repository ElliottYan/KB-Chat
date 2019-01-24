#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2 python3 main_test.py -dec=Tree2Seq -path=save/tree2seq-KVR/HDD128BSZ8DR0.2L1lr0.001Tree2Seq13.05 -bsz=16 -ds=kvr