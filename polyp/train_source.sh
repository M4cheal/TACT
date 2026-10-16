#!/bin/bash
clear
datasets=("BKAI" "CVC-ClinicDB" "ETIS-LaribPolypDB" "Kvasir-SEG")

for dataset in "${datasets[@]}"; do
    echo '------------------------------------------------------------------'
    echo $dataset
    
    /root/miniconda3/envs/CTTAKAN/bin/python \
        /root/code/TACT/polyp/train_source.py \
        --Source_Dataset $dataset --num_epochs 200 \
        --path_save_model "/root/autodl-tmp/save/models/test/polyp_resnet34/" \
        --log_name 'train_Source' --log_label 'train_Source'
    
    echo '------------------------------------------------------------------'
done

