#!/bin/bash


export CUDA_VISIBLE_DEVICES="0"

for sa in 0.01
do
    # echo "sampler No. : ${sa}"
    python run3.py -lr $sa -data 'ml100kdata.mat' -e 200 --sampler 7 -s 200 -b 32 --loss_mode 3 --log_path 'log_sum_log'
done
