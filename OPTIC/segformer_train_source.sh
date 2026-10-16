#!/bin/bash
clear
datasets=("Drishti_GS" "ORIGA" "REFUGE" "REFUGE_Valid" "RIM_ONE_r3")

for dataset in "${datasets[@]}"; do
    echo '------------------------------------------------------------------'
    echo $dataset
    
    /root/miniconda3/envs/CTTAKAN/bin/python \
        /root/code/TACT/OPTIC/segformer_train_source.py \
        --Source_Dataset $dataset --lr 1e-4 --backbone "b0" --num_epochs 100 \
        --log_name "Segformer_b0/train_Source_b0" --log_label "Segformer_b0/train_Source_b0" \
        --path_save_model "/root/autodl-tmp/save/models/test/fundus_segformer_b0/"
    
    echo '------------------------------------------------------------------'
done
