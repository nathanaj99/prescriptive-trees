#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=01:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nathanael.jo@gmail.com

cd /project/vayanou_651/prescriptive-trees/Direct_Approach/


module load gcc
module load gurobi
module load python
export PYTHONPATH=/project/vayanou_651/python/pkgs:${PYTHONPATH}


python run_LAHSA.py -f '../LAHSA/data/disabled_split/FinalData_PresTrees_fixedDisc_rf_Pred.csv' -d '3' -t '36000' -r 'DR' -o '../LAHSA/results/v2/dr_prototype'