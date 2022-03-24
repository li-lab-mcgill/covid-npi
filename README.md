# Inferring global-scale temporal latent topics from news reports to predict public health interventions for COVID-19

[![DOI](https://zenodo.org/badge/353421711.svg)](https://zenodo.org/badge/latestdoi/353421711)

[Published](https://www.cell.com/patterns/fulltext/S2666-3899(22)00002-2) in Patterns. If you find this work helpful, please cite as

```
@article{WEN2022100435,
title = {Inferring global-scale temporal latent topics from news reports to predict public health interventions for COVID-19},
journal = {Patterns},
pages = {100435},
year = {2022},
issn = {2666-3899},
doi = {https://doi.org/10.1016/j.patter.2022.100435},
url = {https://www.sciencedirect.com/science/article/pii/S2666389922000022},
author = {Zhi Wen and Guido Powell and Imane Chafi and David L. Buckeridge and Yue Li},
keywords = {latent topic models, transfer learning, variational autoencoder, non-pharmacological interventions, COVID-19, public health surveillance}
}
```

## Data processing

### 1. AYLIEN

The Python script for AYLIEN data processing is `data_preprocess.py` under the folder `scripts`. The recommended way of using it is running `run_data_process.sh`. This script runs the following command:

```bash
python data_preprocess.py \
    --data_file_path "path to csv file containing news reports and their dates, countries, etc." \
    --stopwords_path "path to stop words file" \
    --cnpi_labels_path "path to file containing country npis" \
    --save_dir "directory to save output" \
    --aylien_flag 1 \
    # --label_harm 1 \  # if harmonize (group) npis
```
`data_preprocess.py` supports processing AYLIEN and WHO data, with `--aylien_flag` set to 1 it would expect to process AYLIEN data. The harmonization of NPIs (grouping all NPIs into 15 groups) is done according to the mapping in `npi labels.xlsx`.

The processed data, e.g. bag-of-words, will be stored in the output directory specified by `save_dir`. This should also be the input directory for running MixMedia (see below).

### 2. WHO

The Python script for WHO data processing is `who_data_process.py` under the folder `scripts`. WHO data should be processed *after* an MixMedia model is trained on AYLIEN data, since it relies on some of its output files (and input files). The recommended way of using the script is running `run_who_data_process.sh`. This script runs the following command:

```bash
python who_data_process.py \
    --who_data_path "path to csv file containing WHO news reports and their dates, countries, etc." \
    --aylien_dir "directory containing the preprocessed AYLIEN data, including mappings of time and source, as well as the vocab" \
    --eta_path "path to the inferred eta file, which sould be in the outputs of running EpiTopics on AYLIEN" \
    --save_dir "directory to save output" \
```

The processed data, including bag-of-words, merged WHO data, and merged country topic priors (etas), will be stored in the output directory specified by `save_dir`. This should also be the input directory for inferring WHO documents' topic mixtures (see below).

## Running MixMedia

The main Python script for training or evaluating MixMedia is `main.py` under `code/MixMedia`, which also contains the supporting scripts. The recommended way of running the model is running `run_MixMedia.sh` under the folder `code`. 

This script has 3 short arguments that are input from command line (number of topics, random seed of the data, GPUs):

```bash
K=$1
seed=$2
cuda=$3
```

There are also longer arguments that are specified within the script:

```bash
dataset="dataset name"
datadir="directory containing processed data files"
outdir="directory to store output"
wemb="word embedding file path"
```

The script runs the following command:

```bash
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
    --multiclass_labels "multiclass or not" \
    --load_from "directory to load checkpoint from (for evaluation)" \
```

The output is saved to a folder under `save_path`: `save_path/{timestamp}`, where the timestamp records the time this script starts to run, and is in the format of `{month}-{day}-{hour}-{minute}`. The trained model will be saved, as well as the learned topics (e.g. $\alpha$, $\eta$, etc.).

The progress can be monitored with Tensorboard or Weights & Biases by setting `logger`.

## Transfer learning for NPI prediction

After MixMedia is trained on AYLIEN, the learned topics can be used for NPI prediction via transfer learning. This consists of three consecutive stages: inferring WHO documents' topic mixtures, training a classifier on document-NPI prediction, transferring the classifier to country-NPI prediction.

### 1. Inferring WHO documents' topic mixtures

This step is done in `infer_theta.py` under `code/transfer_learning`. The recommended way of using it is running `infer_theta.sh` in the same folder. This script has 1 input argument which is for specifying the GPU to use. It runs the following command:

```bash
cuda=$1

CUDA_VISIBLE_DEVICES=${cuda} python infer_theta.py \
    --save_dir "directory to save results" \
    --data_dir "directory containing processed WHO data" \
    --model_dir "the directory containing trained MixMedia model"
```

The output is saved to a folder under `save_dir`: `save_dir/{timestamp}`, where the timestamp records the time this script starts to run, and is in the format of `{month}-{day}-{hour}-{minute}`. The document topic mixtures $\theta$ will be saved.

### 2. Training a classifier on document-NPI prediction

This step is done in `classify_npi.py` under `code/transfer_learning`. The recommended way of using it is running `classify_npi.sh` in the same folder. This script has 1 input argument which is for specifying the GPU to use. It runs the following command:

```bash
cuda=$1

CUDA_VISIBLE_DEVICES=${cuda} python classify_npi.py \
    --mode "running mode (zero-shot, from-scratch, etc.)" \
    --eta_dir "directory containing dynamic topic prior eta (and also alpha)" \
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
    # --save_ckpt \ # save output to save_dir
    # --quiet \ # not showing progress bar
```

Note that `classify_npi.py` and `classify_npi.sh` support all types of NPI predictions, including at the document level, transferring to country-level as well as random baselines. The specific mode of NPI prediction is specified by the `mode` argument. Depending on the mode, some other optional arguments might be required.

For document-level NPI prediction, set `mode` to "doc" and provide `who_label_dir` and `theta_dir`.

Whether the results are saved to disk is controlled by `save_ckpt`. If set, the results are saved to a subfolder under `save_dir`: `save_dir/mode/{timestamp}`, where the timestamp records the time this script starts to run, and is in the format of `{month}-{day}-{hour}-{minute}`. For each random seed, a trained linear classifier and the corresponding test predictions are saved, with suffixes in filenames that specify the seed. The aggregated results in AUPRC are also saved into a json file.

### 3. Transferring the classifier to country-NPI prediction

This step is accomplished in `classify_npi.py` too. See step 2 for details of its usage. For zero-shot transfer, set `mode` to `zero_shot` and provide `cnpi_dir` and `ckpt_dir`; For fine-tuning, set `mode` to `finetune` and provide `cnpi_dir` and `ckpt_dir`.
