## Here is the code for pq_vae


An example run file is *run.py*

Baseline vae_cf   *run.py --model 'vae' --sampler 0*
Our method        *run.py --model 'vae' --sampler {1,2,3}*
sampler 1: uniform 2: popular 3: approximated softmax