# NLGCL: A Contrastive Learning between Neighbor Layers for Graph Collaborative Filtering
## Requirements
```
python>=3.9.18
pytorch>=1.13.1
```

## Dataset

| Datasets  | #Users  | #Items | #Interactions | Sparsity |
| --------- | ------- | ------ | ------------- | -------- |
| Yelp      | 45,477  | 30,708 | 1,777,765     | 99.873%  |
| Pinterest | 55,188  | 9,912  | 1,445,622     | 99.736%  |
| QB-Video  | 30,324  | 25,731 | 1,581,136     | 99.797%  |
| Alibaba   | 300,001 | 81,615 | 1,607,813     | 99.993%  |


## Training
```
python main.py
```