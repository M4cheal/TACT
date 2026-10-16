#!/bin/bash
clear
datasets=("BKAI" "CVC-ClinicDB" "ETIS-LaribPolypDB" "Kvasir-SEG")

for dataset in "${datasets[@]}"; do
    echo '------------------------------------------------------------------'
    echo $dataset
    /root/miniconda3/envs/CTTAKAN/bin/python \
        /root/code/TACT/polyp/TACT.py \
        --Source_Dataset $dataset --sparse_thr 2e-4 --lr 1e-4 \
        --exp_name "TACT/seed0"\
        --model_root "/root/autodl-tmp/save/models/polyp_resnet34/" \
        --alpha_seg 1.0 --alpha_con 0.1 --gamma_con 0.1\
        --thr_t_mask 0.5 --thr_pcl_en 0.75 --thr_pcl_pre 0.5 \
        --lossweight "{'dice':1.0, 'bce':1.0}"\
        --gamma_ent 0.1 --seed 0
    echo '------------------------------------------------------------------'
done
