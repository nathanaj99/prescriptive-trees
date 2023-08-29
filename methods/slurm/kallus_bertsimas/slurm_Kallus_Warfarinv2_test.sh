#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6GB
#SBATCH --time=01:30:00
#SBATCH --export=NONE
#SBATCH --constraint="xeon-2640v4"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nathanael.jo@gmail.com
#SBATCH --array=0-1

cd /project/vayanou_651/prescriptive-trees/Kallus_Bertsimas/


module load gcc
module load gurobi
module load python
export PYTHONPATH=/project/vayanou_651/python/pkgs:${PYTHONPATH}


train_data_list="data_train_enc_0.33_1.csv data_train_enc_0.33_2.csv
"
test_data_list="data_test_enc_0.33_1.csv data_test_enc_0.33_2.csv
"
depth_list="1 1
"
data_group_list="Warfarin_seed1 Warfarin_seed1
"
train_data_list=($train_data_list)
test_data_list=($test_data_list)
depth_list=($depth_list)
data_group_list=($data_group_list)


python main.py -f ${train_data_list[$SLURM_ARRAY_TASK_ID]} -e ${test_data_list[$SLURM_ARRAY_TASK_ID]} -d ${depth_list[$SLURM_ARRAY_TASK_ID]} -t 3600 -b 100 -r kallus -n 0 -g ${data_group_list[$SLURM_ARRAY_TASK_ID]}
