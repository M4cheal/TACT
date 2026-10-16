#!/bin/bash
clear
datasets=("BIDMC" "BMC" "HK" "I2CVB" "RUNMC" "UCL")

for dataset in "${datasets[@]}"; do
    echo '------------------------------------------------------------------'
    echo $dataset
    
    /root/miniconda3/envs/CTTAKAN/bin/python \
        /root/code/TACT/Prostate/baseline_only_source.py \
        --Source_Dataset $dataset \
        --model_root "/root/autodl-tmp/save/models/ProstateSlice/" \
        --exp_name 'Baseline_Only_Source'
    
    echo '------------------------------------------------------------------'
done
