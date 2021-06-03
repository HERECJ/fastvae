## The code and data for FastVAE


An example run file is *run.py*

Baseline vae_cf   
```run.py --model 'vae' --sampler 0```

Our method
```run.py --model 'vae' --sampler No.sampler```

sampler 1: uniform 2: popular 3: Extract 4: MIDX 5: MIDX_Uni 6: MIDX_Pop 7: MIDX_Res 8: DNS 9: Kernel Based  

More parameters can be found in the *run.py* file, including the learning rate, batch size, epoch, step size and so on

The dataset can be found in the 'datasets' directory
