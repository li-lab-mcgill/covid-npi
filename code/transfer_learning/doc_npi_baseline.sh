#!/bin/bash
cuda=$1

CUDA_VISIBLE_DEVICES=${cuda} python doc_npi_baseline.py \
    --data_dir "directory contianing bow" \
    --save_dir "directory to save output" \
    --epochs "# of epochs" \
    --num_seeds "# of random runs" \
    --lr "learning rate" \
    # --save_ckpt \ # save output to save_dir
    # --linear \    # use linear classifier