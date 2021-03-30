#!/bin/bash
cuda=$1

CUDA_VISIBLE_DEVICES=${cuda} python infer_theta.py \
    --save_dir "directory to save results" \
    --data_dir "directory containing processed WHO data" \
    --model_dir "the trained MixMedia model"