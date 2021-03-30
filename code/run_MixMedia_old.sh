#!/bin/bash

K=$1
predict_cnpi=$2
seed=$3
cuda=$4

cd MixMedia_old

dataset="dataset name"
datadir="directory containing processed data files"
outdir="directory to store output"
wemb="word embedding file path"

if [ ! -d $outdir ]; then
	mkdir $outdir -p
fi

CUDA_VISIBLE_DEVICES=${cuda} python main.py \
    --mode "running mode" \
    --dataset ${dataset} \
    --data_path ${datadir} \
    --batch_size "batch size" \
    --emb_path ${wemb} \
    --save_path ${outdir} \
    --lr "learning rate" \
    --epochs "# of epochs" \
    --num_topics ${K} \
    --min_df "minimal document frequency (this matches with the data processing)" \
    --train_embeddings "train word embeddings or not" \
    --eval_batch_size "evaluation batch size" \
    --time_prior "use time-varying prior or not" \
    --source_prior "use country-specific prior or not" \
    --logger "which logger to use" \
    --predict_cnpi ${predict_cnpi} \
    --multiclass_labels "multiclass or not" \
    --load_from "directory to load checkpoint from (for evaluation)" \
