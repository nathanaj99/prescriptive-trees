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


python run_LAHSA.py -f '../LAHSA/data/pt_input_prob_downsampled_5000.csv' -d '3' -t '36000' -r 'DM' -o '../LAHSA/results/dm_downsampled_5000.lp'