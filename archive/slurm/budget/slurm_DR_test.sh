#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --constraint="xeon-2640v4"
#SBATCH --array=0-1

cd /project/vayanou_651/prescriptive-trees/Direct_Approach/


module load gcc/11.3.0
module load gurobi/10.0.1
module load python/3.6.5
export PYTHONPATH=/project/vayanou_651/python/pkgs:${PYTHONPATH}


train_data_list="data_train_enc_0.5_1.csv data_train_enc_0.5_2.csv
"
test_data_list="data_test_enc_0.5_1.csv data_test_enc_0.5_2.csv
"
depth_list="1 1
"
pred_type_list="true true
"
data_group_list="Athey_v1_500 Athey_v1_500
"
ml_list="linear linear
"
budgets_list="0.25 0.25
"
train_data_list=($train_data_list)
test_data_list=($test_data_list)
depth_list=($depth_list)
pred_type_list=($pred_type_list)
data_group_list=($data_group_list)
ml_list=($ml_list)
budgets_list=($budgets_list)


python main_synthetic.py -f ${train_data_list[$SLURM_ARRAY_TASK_ID]} -e ${test_data_list[$SLURM_ARRAY_TASK_ID]} -d ${depth_list[$SLURM_ARRAY_TASK_ID]} -t 3600 -b 100 -p ${pred_type_list[$SLURM_ARRAY_TASK_ID]} -r robust -g ${data_group_list[$SLURM_ARRAY_TASK_ID]} -m ${ml_list[$SLURM_ARRAY_TASK_ID]} -u ${budgets_list[$SLURM_ARRAY_TASK_ID]}
