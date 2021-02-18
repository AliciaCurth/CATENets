# CATENets - Conditional Average Treatment Effect Estimation Using Neural Networks
Code Author: Alicia Curth

This repo contains Jax-based, sklearn-style implementations of Neural Networ-based Conditional Average Treatment Effect (CATE) Estimators, which were used in the AISTATS 2021 paper 'Nonparametric Estimation of Heterogeneous Treatment Effects: From Theory to Learning Algorithms' (https://arxiv.org/abs/2101.10943).

We implement the SNet-class we introduce in our AISTATS paper, and re-implement a number of NN-based algorithms from existing literature (Shalit et al (2017), Shi et al (2019), Hassanpour & Greiner (2020)). We also consider NN-based instantiations of a number of two-step pseudo-regression estimators, including the DR-learner (Kennedy, 2020) and single-robust propensity-weighted and regression-adjusted learners. 

### Citing 

If you use this software please cite as follows:

```
@inproceedings{curth2021nonparametric,
  title={Nonparametric Estimation of Heterogeneous Treatment Effects: From Theory to Learning Algorithms},
  author={Curth, Alicia and van der Schaar, Mihaela}
    year={2021},
     booktitle={Proceedings of the 24th International Conference on Artificial
  Intelligence and Statistics (AISTATS)},
    url={https://arxiv.org/abs/2101.10943}
}
```

