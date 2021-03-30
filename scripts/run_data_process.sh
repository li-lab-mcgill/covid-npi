#!/bin/bash

python data_preprocess.py \
    --data_file_path "path to csv file containing news reports and their dates, countries, etc." \
    --stopwords_path "path to stop words file" \
    --cnpi_labels_path "path to file containing country npis" \
    --save_dir "directory to save output" \
    # --aylien_flag 1 \ # if processing aylien data
    # --label_harm 1 \  # if harmonize (group) npis