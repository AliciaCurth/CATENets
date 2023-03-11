# Replication code for "Really Doing Great at Estimating CATE? A Critical Look at ML Benchmarking Practices in Treatment Effect Estimation"

This folder contains the files to replicate the benchmarking studies of random forest (RF) and neural network (NN) based CATE estimators using the IHDP, ACIC2016 and Twins datasets.

The code for RFs is in R and relies on the R-package ‘grf’ which is available on CRAN. The code for NNs relies on the python package ‘catenets’ in this repo.

This folder provides both python and R code to replicate the results of all empirical studies. Always run the python code first, this code downloads and/or creates the datasets that are used in both python and R code.
The python code can also be run using the file 'run_experiments_benchmarks_NeurIPS.py' in the root of the repo.

For IHDP: Setting ‘original’ reproduces results as reported in Figure 3a and setting ‘modified’ reproduces results in Figure 3b.

For ACIC: we considered the simulation numbers (`simu_num’) 2, 26 and 7.
