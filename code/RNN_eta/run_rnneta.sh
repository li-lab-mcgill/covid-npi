#!/bin/bash

seed=$1
cuda=$2
eta_path=$3

dataset=WHO_910_harm
datadir=/home/mcb/users/zwen8/data/covid/${dataset}_cnpi_${seed}/
outdir=/home/mcb/users/zwen8/data/covid/results/cnpi/${dataset}/rnn_eta/seed_${seed}

if [ ! -d $outdir ]; then
	mkdir $outdir -p
fi

CUDA_VISIBLE_DEVICES=${cuda} python main.py \
    --dataset ${dataset} \
    --data_path ${datadir} \
    --eta_path ${eta_path} \
    --save_path ${outdir} \
    --seed ${seed} \
    --num_layers 1 \
    --hidden_size 512 \
    --batch_size 128 \
    --lr 1e-3 \
    --embed_topic_with_alpha \