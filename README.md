# NLGCL: A Contrastive Learning between Neighbor Layers for Graph Collaborative Filtering

## News
```
An extended version [NLGCL+](https://github.com/Jinfeng-Xu/NLGCL-Plus) focused on extending to multimodal recommendation scenarios was accepted by TORS 2026.
```

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

If you find NLGCL useful in your research, please consider citing our [NLGCL paper](https://arxiv.org/pdf/2507.07522).
If you find NLGCL+ useful in your research, please consider citing our [NLGCL+ paper](https://dl.acm.org/doi/pdf/10.1145/3806231). 

```
1. NLGCL (RecSys2025 Spotlight Oral)
@inproceedings{xu2025nlgcl,
  title={NLGCL: Naturally Existing Neighbor Layers Graph Contrastive Learning for Recommendation},
  author={Xu, Jinfeng and Chen, Zheyu and Yang, Shuo and Li, Jinze and Wang, Hewei and Wang, Wei and Hu, Xiping and Ngai, Edith},
  booktitle={Proceedings of the Nineteenth ACM Conference on Recommender Systems},
  pages={319--329},
  year={2025}
}

2. NLGCL+ (TORS2026)
@article{xu2026nlgcl+,
  title={NLGCL+: Naturally Existing Neighbour Layers Graph Contrastive Learning with Adaptive Sample Weighting for Multimodal Recommendation},
  author={Xu, Jinfeng and Chen, Zheyu and Yang, Shuo and Li, Jinze and Wang, Hewei and Wang, Wei and Hu, Xiping and Ngai, Edith},
  journal={ACM Transactions on Recommender Systems},
  year={2026},
  publisher={ACM New York, NY}
}
```