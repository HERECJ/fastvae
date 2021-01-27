# Experiment List

## Dataset
Filter user and item which has less than 10 interactions
Implicit feedbacks, if the user interacted the item, denoted as 1 else 0
Data format: **{data_name}.mat**, usually the sparse matrix
+ MovieLens100K(test)
+ MovieLens10M
+ Yelp
+ Amazon

## Baselines
+ **VAE-CF [Variational Autoencoders for Collaborative Filtering]** the highest prior
+ NeuHash-CF [Content-aware Neural Hashing for Cold-start Recommendation]
+ Other sampling method
<!-- + NCF [Neural collaborative filtering] -->

**Our Methods**
+ No quantized vectors/ Real values of vectors (vae, user_quatized=False) **Prior**
+ Quantized User Embeddings

## Baseline Experiment
### Parameter Settings
+ embedding_dim : 64
+ split_ratio : 0.8
+ Optimizer : Adam
+ Num_sample : 5
+ Learning_rate : tuning
+ Other parameters for tuning our methods: Cluster_num ...

### Metric
+ NACG @ 5, 10, 50
+ RECALL @ 5, 10, 50

## Sensitive Analysis
For sampling
+ The number of sampled negtive items : [5, 10, 20, 50]
+ The number of clusters for sampling [16, 32, 64, 128, 256]
+ The dimensionn of centers : fix the item embedding pick 4~5 values

Other
+ The dimension of embeddings : choose the best baseline algorithm