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

## Citing NLGCL

If you find NLGCL useful in your research, please consider citing our [paper](https://arxiv.org/pdf/2507.07522).

```
@article{xu2025nlgcl,
  title={NLGCL: Naturally Existing Neighbor Layers Graph Contrastive Learning for Recommendation},
  author={Xu, Jinfeng and Chen, Zheyu and Yang, Shuo and Li, Jinze and Wang, Hewei and Wang, Wei and Hu, Xiping and Ngai, Edith},
  journal={arXiv preprint arXiv:2507.07522},
  year={2025}
}
```
