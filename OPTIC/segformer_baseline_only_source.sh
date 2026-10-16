#!/bin/bash
clear
datasets=("Drishti_GS" "ORIGA" "REFUGE" "REFUGE_Valid" "RIM_ONE_r3")

for dataset in "${datasets[@]}"; do
    echo '------------------------------------------------------------------'
    echo $dataset
    
    /root/miniconda3/envs/CTTAKAN/bin/python \
        /root/code/TACT/OPTIC/segformer_baseline_only_source.py \
        --Source_Dataset $dataset --backbone "b0" \
        --exp_name 'Segformer_b0/Baseline_Only_Source' \
        --model_root "/root/autodl-tmp/save/models/fundus_segformer_b0/"
    
    echo '------------------------------------------------------------------'
done
