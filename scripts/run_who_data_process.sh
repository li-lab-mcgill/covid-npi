#!/bin/bash

python who_data_process.py \
    --who_data_path "path to csv file containing WHO news reports and their dates, countries, etc." \
    --aylien_dir "directory containing the preprocessed AYLIEN data, including mappings of time and source, as well as the vocab" \
    --eta_path "path to the inferred eta file, which sould be in the outputs of running EpiTopics on AYLIEN" \
    --save_dir "directory to save output" \