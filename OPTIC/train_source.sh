#!/bin/bash
clear
datasets=("Drishti_GS" "ORIGA" "REFUGE" "REFUGE_Valid" "RIM_ONE_r3")

for dataset in "${datasets[@]}"; do
    echo '------------------------------------------------------------------'
    echo $dataset
    
    /root/miniconda3/envs/CTTAKAN/bin/python \
        /root/code/TACT/OPTIC/train_source.py \
        --Source_Dataset $dataset \
        --log_name "ResUnet/train_Source" --log_label "ResUnet/train_Source" \
        --path_save_model "/root/autodl-tmp/save/models/test/fundus/"
    
    echo '------------------------------------------------------------------'
done
