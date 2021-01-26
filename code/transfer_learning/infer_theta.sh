#!/bin/bash
cuda=$1

CUDA_VISIBLE_DEVICES=${cuda} python infer_theta.py \
    --save_dir /home/mcb/users/zwen8/data/covid/results/WHO_910_transfer_harm_0 \
    --data_dir /home/mcb/users/zwen8/data/covid/WHO_910_transfer_harm_0/min_df_10 \
    --model_dir /home/mcb/users/zwen8/data/covid/results/cnpi/Aylien_july_cdc_harm/old/lstm/seed_0/01-05-15-09/