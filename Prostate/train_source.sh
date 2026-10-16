#!/bin/bash
clear
datasets=("BIDMC" "BMC" "HK" "I2CVB" "RUNMC" "UCL")

for dataset in "${datasets[@]}"; do
    echo '------------------------------------------------------------------'
    echo $dataset
    
    /root/miniconda3/envs/CTTAKAN/bin/python \
        /root/code/TACT/Prostate/train_source.py \
        --Source_Dataset $dataset --num_epochs 100 \
        --path_save_model "/root/autodl-tmp/save/models/test/ProstateSlice/" \
        --log_name 'train_Source' --log_label 'train_Source'
    
    echo '------------------------------------------------------------------'
done