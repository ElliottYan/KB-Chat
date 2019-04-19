#!/usr/bin/env bash

python process_data_kvr.py --json=kvret_train_public.json> data/KVR/kvr_train.txt
python process_data_kvr.py --json=kvret_dev_public.json > data/KVR/kvr_dev.txt
python process_data_kvr.py --json=kvret_test_public.json > data/KVR/kvr_test.txt
