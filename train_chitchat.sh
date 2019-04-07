#!/usr/bin/env bash
bsz=256
lr=0.001
n_layer=1
hdd=512
dec=Mem2SeqChitChat
dropout=0.2
ds=chitchat

CUDA_VISIBLE_DEVICES=2 python3 main_train.py -lr=$lr -layer=$n_layer -hdd=$hdd -dr=$dropout -dec=$dec -bsz=$bsz -ds=$ds -task=chitchat -t= | tee log/${ds}_"0.3,0.3,0"_${dec}_${hdd}_${n_layer}_${bsz}_${dropout}_${lr}.log