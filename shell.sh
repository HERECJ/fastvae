#!/bin/bash


# for i in 0.001 0.0001
# do 
#     echo "lr: ${i}";
#     python run.py -lr $i -data 'ml100kdata.mat' -e 150 --sampler 3 -s 100 ;
# done
export CUDA_VISIBLE_DEVICES="0"
# for s in 10 50 100 200 500 1000
# do
#     echo "num_sampled : ${s}";
#     # python run.py -lr 0.1 -data 'ml100kdata.mat' -e 100 --sampler 3 -s $s -b 32;
#     python run.py -lr 0.1 -data 'ml100kdata.mat' -e 100 --sampler 1 -s $s -b 32 --loss_mode 2;
# done
for sa in 0.01
do
    # echo "sampler No. : ${sa}"
    python run3.py -lr $sa -data 'ml100kdata.mat' -e 200 --sampler 7 -s 200 -b 32 --loss_mode 3 --log_path 'log_sum_log'
done

# python run2.py -lr 0.1 -data 'ml100kdata.mat' -e 200 --sampler 4 -s 400 -b 32 --loss_mode 3 --log_path 'log_base'