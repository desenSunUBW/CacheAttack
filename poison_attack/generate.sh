#!/bin/bash

# Simple script to run Python prompt processor 6 times with different output paths
# PROMPT_FILE="$1"
# LOGOs=("blue moon sign" "Mcdonald sign" "Apple sign" "Chanel symbol" "circled triangle symbol" "circled Nike symbol")
LOGOs=("Apple sign")
MODEL="flux"
BASIC_PATH=""
DATASET="lexica"
for i in "${!LOGOs[@]}"; do
    python3 generate_images_with_logo.py "${BASIC_PATH}/${LOGOs[i]}/${DATASET}-${MODEL}_LCBFU_1_cachetox_hit_rate.csv" --output-dir "${BASIC_PATH}/${LOGOs[i]}/images/${DATASET}/${MODEL}" --generate-cache --model "${MODEL}" --cache-dir "${BASIC_PATH}/${LOGOs[i]}/cache/${DATASET}/${MODEL}" --cache-path "${BASIC_PATH}/${LOGOs[i]}/${DATASET}-${MODEL}_cachetox.log"
done