#!/bin/bash
clear
datasets=("Drishti_GS" "ORIGA" "REFUGE" "REFUGE_Valid" "RIM_ONE_r3")

for dataset in "${datasets[@]}"; do
    echo '------------------------------------------------------------------'
    echo $dataset
    
    /root/miniconda3/envs/CTTAKAN/bin/python \
        /root/code/TACT/OPTIC/baseline_only_source.py \
        --Source_Dataset $dataset \
        --exp_name 'ResUnet/Baseline_Only_Source'
    
    echo '------------------------------------------------------------------'
done

