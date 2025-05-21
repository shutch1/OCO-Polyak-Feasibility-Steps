# Code for "Constrained Online Convex Optimization with Polyak Feasibility Steps"


This repository contains the code for the following paper:

> Hutchinson, Alizadeh, "Constrained Online Convex Optimization with Polyak Feasibility Steps". International Conference on Machine Learning (ICML) 2025.

The arXiv version of this paper can be found at [arXiv:2502.13112](https://arxiv.org/abs/2502.13112).

The following instructions show how to generate the experiment results shown in the paper.

## Setting Up Environment

Before using the code, the conda environment needs to be setup by navigating to the repository directory and then using the following command:
```
conda env create -f environment.yml
```
The environment then needs to be activated with the following command:
```
conda activate const_oco
```

## Running Experiments

To generate the figures in the paper, one needs to first run the appropriate codes. The files corresponding to each plot are:
- **Figure 1a:** [dpp/reg_dpp.py](dpp/reg_dpp.py), [dppt/reg_dpp_tight.py](dppt/reg_dpp_tight.py), [pfs/reg_dpp.py](pfs/reg_pfs.py)
- **Figure 1b:** [dpp/reg_dpp.py](dpp/reg_dpp.py), [dppt/reg_dpp_tight.py](dppt/reg_dpp_tight.py), [pfs/reg_dpp.py](pfs/reg_pfs.py)
- **Figure 1c:** [dpp/viol_dpp.py](dpp/viol_dpp.py), [dppt/viol_dpp_tight.py](dppt/viol_dpp_tight.py), [pfs/viol_dpp.py](pfs/viol_pfs.py)
- **Figure 2:** [dpp/act_dpp.py](dpp/act_dpp.py), [pfs/act_pfs.py](pfs/act_pfs.py)

## Creating Plots

The plots can be recreated and saved by running all of the cells in the Jupyter notebook file [plot_data.ipynb](plot_data.ipynb).
