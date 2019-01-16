#!/usr/bin/env bash

python3 main_train.py -lr=0.001 -layer=1 -hdd=128 -dr=0.2 -dec=Tree2Seq -bsz=8 -ds=kvr -task=kvr -t=