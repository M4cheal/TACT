#!/bin/bash
clear
datasets=("BKAI" "CVC-ClinicDB" "ETIS-LaribPolypDB" "Kvasir-SEG")

for dataset in "${datasets[@]}"; do
    echo '------------------------------------------------------------------'
    echo $dataset
    
    /root/miniconda3/envs/CTTAKAN/bin/python \
        /root/code/TACT/polyp/baseline_only_source.py \
        --Source_Dataset $dataset \
        --model_root "/root/autodl-tmp/save/models/polyp_resnet34/" \
        --exp_name 'Baseline_Only_Source'
    
    echo '------------------------------------------------------------------'
done

