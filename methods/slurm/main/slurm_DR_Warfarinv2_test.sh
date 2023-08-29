#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6GB
#SBATCH --time=04:15:00
#SBATCH --export=NONE
#SBATCH --constraint="xeon-2640v4"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nathanael.jo@gmail.com
#SBATCH --array=0-1

cd /project/vayanou_651/prescriptive-trees/Direct_Approach/


module load gcc
module load gurobi
module load python
export PYTHONPATH=/project/vayanou_651/python/pkgs:${PYTHONPATH}


train_data_list="data_train_enc_r0.06_1.csv data_train_enc_r0.11_1.csv
"
test_data_list="data_test_enc_r0.06_1.csv data_test_enc_r0.11_1.csv
"
depth_list="2 2
"
pred_type_list="tree tree
"
data_group_list="Warfarin_seed1 Warfarin_seed1
"
ml_list="lrrf lrrf
"
treatment_options_list="3 3
"
train_data_list=($train_data_list)
test_data_list=($test_data_list)
depth_list=($depth_list)
pred_type_list=($pred_type_list)
data_group_list=($data_group_list)
ml_list=($ml_list)
treatment_options_list=($treatment_options_list)


python main_agg.py -f ${train_data_list[$SLURM_ARRAY_TASK_ID]} -e ${test_data_list[$SLURM_ARRAY_TASK_ID]} -d ${depth_list[$SLURM_ARRAY_TASK_ID]} -t 14400 -b 100 -p ${pred_type_list[$SLURM_ARRAY_TASK_ID]} -r robust -g ${data_group_list[$SLURM_ARRAY_TASK_ID]} -m ${ml_list[$SLURM_ARRAY_TASK_ID]} -u ${treatment_options_list[$SLURM_ARRAY_TASK_ID]}
