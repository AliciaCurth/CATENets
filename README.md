# CATENets - Conditional Average Treatment Effect Estimation Using Neural Networks
Code Author: Alicia Curth (amc253@cam.ac.uk)

This repo contains Jax-based, sklearn-style implementations of Neural Network-based Conditional 
Average Treatment Effect (CATE) Estimators, which were used in the AISTATS 2021 paper 
'Nonparametric Estimation of Heterogeneous Treatment Effects: From Theory to Learning 
Algorithms' (Curth & vd Schaar, 2021a; https://arxiv.org/abs/2101.10943) as well as the follow up 
paper "On Inductive Biases for Heterogeneous Treatment Effect Estimation" (Curth & vd 
Schaar, 2021b; https://arxiv.org/abs/2106.03765).

We implement the SNet-class we introduce in Curth & vd Schaar (2021a), as well as FlexTENet and 
OffsetNet as discussed in Curth & vd Schaar (2021b), and re-implement a number of 
NN-based algorithms from existing literature (Shalit et al (2017), Shi et al (2019), Hassanpour 
& Greiner (2020)). We also provide NN-based instantiations of a number of so-called 
meta-learners for CATE estimation, including two-step pseudo-outcome regression estimators (the 
DR-learner (Kennedy, 2020) and single-robust propensity-weighted (PW) and regression-adjusted  
(RA) learners), Nie & Wager (2017)'s R-learner and Kuenzel et al (2019)'s X-learner. 

### Interface
All implemented learning algorithms (``SNet, FlexTENet, OffsetNet, TNet, SNet1 (TARNet), SNet2 
(DragonNet), SNet3, DRNet, RANet, PWNet, RNet, XNet``) come with a sklearn-style wrapper,  
implementing a ``.fit(X, y, w)`` and a ``.predict(X)`` method, where predict returns CATE by  
default. All hyperparameters are documented in detail in the respective files in the .models folder.

Example usage:

```python
from catenets.models import TNet, SNet
from catenets.experiment_utils.simulation_utils import simulate_treatment_setup

# simulate some data (here: unconfounded, 10 prognostic variables and 5 predictive variables)
X, y, w, p, cate = simulate_treatment_setup(n=2000, n_o=10, n_t=5, n_c=0)

# estimate CATE using TNet
t = TNet()
t.fit(X, y, w)
cate_pred_t = t.predict(X)  # without potential outcomes
cate_pred_t, po0_pred_t, po1_pred_t = t.predict(X, return_po=True)  # predict potential outcomes too

# estimate CATE using SNet
s = SNet(penalty_orthogonal=0.01)
s.fit(X, y, w)
cate_pred_s = s.predict(X)

```

All experiments in Curth & vd Schaar (2021a) can be replicated using this repository; the necessary 
code is in ``experiments.experiments_AISTATS21``. To do so from shell, clone the repo, create a new 
virtual environment and run
```shell
pip install -r requirements.txt #install requirements
python run_experiments_AISTATS.py 
```
```shell
Options:
--experiment # defaults to 'simulation', 'ihdp' will run ihdp experiments
--setting # different simulation settings in synthetic experiments (can be 1-5)
--models # defaults to None which will train all models considered in paper, 
         # can be string of model name (e.g 'TNet'), 'plug' for all plugin models,
         # 'pseudo' for all pseudo-outcome regression models
--file_name # base file name to write to, defaults to 'results'
--n_repeats # number of experiments to run for each configuration, defaults to 10 (should be set to 100 for IHDP)
```
To run the ihdp experiments, first download the IHDP-100 data files from https://www.fredjo.com/ and place them in a folder called 'data/'. 

Similarly, the experiments in Curth & vd Schaar (2021b) can be replicated using the code in 
``experiments.experiments_inductive_bias`` or from shell using ```python 
run_experiments_inductive_bias.py```.

The code can also be installed as a python package (``catenets``). From a local copy of the repo, run ``python setup.py install``. 

Note: jax is currently only supported on macOS and linux, but can be run from windows using WSL (the windows subsystem for linux). 


### Citing 

If you use this software please cite the corresponding paper(s):

```
@inproceedings{curth2021nonparametric,
  title={Nonparametric Estimation of Heterogeneous Treatment Effects: From Theory to Learning Algorithms},
  author={Curth, Alicia and van der Schaar, Mihaela},
    year={2021},
     booktitle={Proceedings of the 24th International Conference on Artificial
  Intelligence and Statistics (AISTATS)},
  organization={PMLR}
}

@article{curth2021inductive,
  title={On Inductive Biases for Heterogeneous Treatment Effect Estimation},
  author={Curth, Alicia and van der Schaar, Mihaela},
  journal={arXiv preprint arXiv:2106.03765},
  year={2021}
}
```

