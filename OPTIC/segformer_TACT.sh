#!/bin/bash
clear
datasets=("Drishti_GS" "ORIGA" "REFUGE" "REFUGE_Valid" "RIM_ONE_r3")

for dataset in "${datasets[@]}"; do
    echo '------------------------------------------------------------------'
    echo $dataset
    /root/miniconda3/envs/CTTAKAN/bin/python \
        /root/code/TACT/OPTIC/segformer_TACT.py \
        --Source_Dataset $dataset --sparse_thr 6e-4 --lr 1e-5 \
        --exp_name "Segformer_b0/TACT/seed0" \
        --alpha_seg 1.0 --alpha_con 0.1 --gamma_con 0.1\
        --thr_t_mask 0.5 --thr_pcl_en 0.75 --thr_pcl_pre 0.5 \
        --lossweight "{'dice':1.0, 'bce':1.0}"\
        --gamma_ent 0.1 --seed 0 \
        --backbone "b0" --model_root "/root/autodl-tmp/save/models/fundus_segformer_b0/"
    echo '------------------------------------------------------------------'
done
