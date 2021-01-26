#!/bin/bash
cuda=$1

CUDA_VISIBLE_DEVICES=${cuda} python doc_npi_baseline.py \
    --data_dir /home/mcb/users/zwen8/data/covid/WHO_910_transfer_harm_0/min_df_10 \
    --save_dir /home/mcb/users/zwen8/data/covid/results/WHO_910_transfer_harm_0 \
    --epochs 10 \
    --num_seeds 100 \
    --lr 1e-4 \
    --no_logger \
    # --save_ckpt \