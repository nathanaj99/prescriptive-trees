#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=10:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nathanael.jo@gmail.com

cd /project/vayanou_651/prescriptive-trees/Direct_Approach/


module load gcc
module load gurobi
module load python
export PYTHONPATH=/project/vayanou_651/python/pkgs:${PYTHONPATH}


python run_LAHSA.py -f '../LAHSA/data/prescriptive_trees_input_prob.csv' -d '2' -t '36000' -r 'IPW' -o '../LAHSA/results/dm_downsampled_5000.lp'