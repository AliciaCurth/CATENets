# CATENets - Conditional Average Treatment Effect Estimation Using Neural Networks
Code Author: Alicia Curth

This repo contains Jax-based, sklearn-style implementations of Neural Networ-based Conditional Average Treatment Effect (CATE) Estimators, which were used in the AISTATS 2021 paper 'Nonparametric Estimation of Heterogeneous Treatment Effects: From Theory to Learning Algorithms' (https://arxiv.org/abs/2101.10943).

We implement the SNet-class we introduce in our AISTATS paper, and re-implement a number of NN-based algorithms from existing literature (Shalit et al (2017), Shi et al (2019), Hassanpour & Greiner (2020)). We also consider NN-based instantiations of a number of two-step pseudo-regression estimators, including the DR-learner (Kennedy, 2020) and single-robust propensity-weighted and regression-adjusted learners. 

### Interface
All learning algorithms are implemented with a sklearn-style wrapper, implementing a ``.fit(X, y, w)`` and a ``.predict(X)`` method, where predict returns CATE by default. 

Example usage:

```python
from catenets.models import TNet, SNet
from catenets.experiments.simulation_utils import simulate_treatment_setup

# simulate some data (here: unconfounded, 10 prognostic variables and 5 predictive variables)
X, y, w, p, cate = simulate_treatment_setup(n=1000, n_o=10, n_t=5, n_c=0)

# estimate CATE using TNet
t = TNet()
t.fit(X, y, w)
cate_pred_t = t.predict(X) # without potential outcomes
cate_pred_t, po0_pred_t, po1_pred_t = t.predict(X, return_po=True) # predict potential outcomes too

# estimate CATE using SNet
s = SNet(penalty_orthogonal=0.01)
s.fit(X, y, w)
cate_pred_s = s.predict(X)

```

All experiments in the AISTATS paper can be replicated using this repository; the necessary code is in ``catenets.experiments.simulations_AISTATS`` and ``catenets.experiments.ihdp_experiments``. 

### Installing
The code is setup as a python package and can by installed by cloning the repository, creating a new virtual environment and running
```shell
pip install -r requirements.txt
python setup.py install
```
Note: jax is currently only supported on macOS and linux, but can be run from windows using WSL (the windows subsystem for linux). 


### Citing 

If you use this software please cite as follows:

```
@inproceedings{curth2021nonparametric,
  title={Nonparametric Estimation of Heterogeneous Treatment Effects: From Theory to Learning Algorithms},
  author={Curth, Alicia and van der Schaar, Mihaela}
    year={2021},
     booktitle={Proceedings of the 24th International Conference on Artificial
  Intelligence and Statistics (AISTATS)}
}
```

