#!/bin/bash
cuda=$1

CUDA_VISIBLE_DEVICES=${cuda} python classify_npi.py \
    --mode "running mode (zero-shot, from-scratch, etc.)" \
    --eta_dir "directory containing dynamic topic prior eta (and also alpha and theta)" \
    --theta_dir "optional, directory containing document topic mixture theta" \
    --who_label_dir "optional, directory containing who labels" \
    --cnpi_dir "optional, directory containing country npis" \
    --ckpt_dir "optional, directory to load checkpoints from (for finetuning and zero-shot)" \
    --save_dir "optional, root directory to save output (models, results, test predictions etc.)" \
    --num_seeds "# of random runs, default 1" \
    --lr "learning rate" \
    --epochs "# of training epochs" \
    --batch_size "batch size" \
    --weight_decay "weight decay for adam optimizer" \
    # --save_ckpt \ save output to save_dir
    # --quiet \ not showing progress bar