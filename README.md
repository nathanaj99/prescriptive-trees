# Code for "Learning Optimal Prescriptive Trees from Observational Data"
### https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4355984 # REPLACE


# Data
We run experiments on a synthetic dataset and a real dataset on warfarin dosing. For the synthetic data, run the scripts in data/datagen/synthetic/ in the order specified. For the warfarin data, run the scripts in data/datagen/warfarin/ in the order specified. The raw warfarin data can be found in XX [LINK HERE]

# Methods
The following describes the structure of our scripts for various methods:
- OPT -- our proposed methods IPW, DM, and DR
	- Note that main.py and Primal.py are now deprecated and we rely on the faster main_agg.py and Primal_agg.py instead
	- All other scripts contain helper functions and classes
- Kallus_Bertsimas -- the methods of Kallus (2017) and Bertsimas et al. (2019), which correspond to K-PT and B-PT in the paper
	- main.py and Primal.py are the main scripts, all others are helper functions/classes 
- CF -- causal forests and causal trees as proposed by Athey et al. (2016)
- policytree -- method proposed by Zhou et al. (2022)
- RC -- regress and compare approach as described in the paper

The folder slurm/ contains bash files to run our methods (OPT), K-PT, and B-PT using slurm.

# Visualization
In analysis_viz/, scripts are named based on the figure or table to which it correponds in the paper. Note that we publish our raw results in XX [DROPBOX LINK?], so running the methods from scratch is not strictly necessary.

DEPENDENCIES?


# R Packages needed
The following R packages are required to run the .R scripts:
- grf
- policytree
- dplyr
- plyr
- tictoc
- glue
- ggplot2

# Python Packages needed
The following python packages are required to run the .py scripts:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imblearn