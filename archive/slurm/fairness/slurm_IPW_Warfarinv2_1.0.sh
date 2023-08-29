#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6GB
#SBATCH --time=01:30:00
#SBATCH --export=NONE
#SBATCH --constraint="xeon-2640v4"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nathanael.jo@gmail.com
#SBATCH --array=0-74
#SBATCH --dependency=afterany:14391024

cd /project/vayanou_651/prescriptive-trees/IPW/


module load gcc
module load gurobi
module load python
export PYTHONPATH=/project/vayanou_651/python/pkgs:${PYTHONPATH}


train_data_list="data_train_enc_0.33_1.csv data_train_enc_0.33_1.csv data_train_enc_0.33_1.csv data_train_enc_0.33_1.csv data_train_enc_0.33_1.csv data_train_enc_r0.06_1.csv data_train_enc_r0.06_1.csv data_train_enc_r0.06_1.csv data_train_enc_r0.06_1.csv data_train_enc_r0.06_1.csv data_train_enc_r0.11_1.csv data_train_enc_r0.11_1.csv data_train_enc_r0.11_1.csv data_train_enc_r0.11_1.csv data_train_enc_r0.11_1.csv data_train_enc_0.33_2.csv data_train_enc_0.33_2.csv data_train_enc_0.33_2.csv data_train_enc_0.33_2.csv data_train_enc_0.33_2.csv data_train_enc_r0.06_2.csv data_train_enc_r0.06_2.csv data_train_enc_r0.06_2.csv data_train_enc_r0.06_2.csv data_train_enc_r0.06_2.csv data_train_enc_r0.11_2.csv data_train_enc_r0.11_2.csv data_train_enc_r0.11_2.csv data_train_enc_r0.11_2.csv data_train_enc_r0.11_2.csv data_train_enc_0.33_3.csv data_train_enc_0.33_3.csv data_train_enc_0.33_3.csv data_train_enc_0.33_3.csv data_train_enc_0.33_3.csv data_train_enc_r0.06_3.csv data_train_enc_r0.06_3.csv data_train_enc_r0.06_3.csv data_train_enc_r0.06_3.csv data_train_enc_r0.06_3.csv data_train_enc_r0.11_3.csv data_train_enc_r0.11_3.csv data_train_enc_r0.11_3.csv data_train_enc_r0.11_3.csv data_train_enc_r0.11_3.csv data_train_enc_0.33_4.csv data_train_enc_0.33_4.csv data_train_enc_0.33_4.csv data_train_enc_0.33_4.csv data_train_enc_0.33_4.csv data_train_enc_r0.06_4.csv data_train_enc_r0.06_4.csv data_train_enc_r0.06_4.csv data_train_enc_r0.06_4.csv data_train_enc_r0.06_4.csv data_train_enc_r0.11_4.csv data_train_enc_r0.11_4.csv data_train_enc_r0.11_4.csv data_train_enc_r0.11_4.csv data_train_enc_r0.11_4.csv data_train_enc_0.33_5.csv data_train_enc_0.33_5.csv data_train_enc_0.33_5.csv data_train_enc_0.33_5.csv data_train_enc_0.33_5.csv data_train_enc_r0.06_5.csv data_train_enc_r0.06_5.csv data_train_enc_r0.06_5.csv data_train_enc_r0.06_5.csv data_train_enc_r0.06_5.csv data_train_enc_r0.11_5.csv data_train_enc_r0.11_5.csv data_train_enc_r0.11_5.csv data_train_enc_r0.11_5.csv data_train_enc_r0.11_5.csv
"
test_data_list="data_test_enc_0.33_1.csv data_test_enc_0.33_1.csv data_test_enc_0.33_1.csv data_test_enc_0.33_1.csv data_test_enc_0.33_1.csv data_test_enc_r0.06_1.csv data_test_enc_r0.06_1.csv data_test_enc_r0.06_1.csv data_test_enc_r0.06_1.csv data_test_enc_r0.06_1.csv data_test_enc_r0.11_1.csv data_test_enc_r0.11_1.csv data_test_enc_r0.11_1.csv data_test_enc_r0.11_1.csv data_test_enc_r0.11_1.csv data_test_enc_0.33_2.csv data_test_enc_0.33_2.csv data_test_enc_0.33_2.csv data_test_enc_0.33_2.csv data_test_enc_0.33_2.csv data_test_enc_r0.06_2.csv data_test_enc_r0.06_2.csv data_test_enc_r0.06_2.csv data_test_enc_r0.06_2.csv data_test_enc_r0.06_2.csv data_test_enc_r0.11_2.csv data_test_enc_r0.11_2.csv data_test_enc_r0.11_2.csv data_test_enc_r0.11_2.csv data_test_enc_r0.11_2.csv data_test_enc_0.33_3.csv data_test_enc_0.33_3.csv data_test_enc_0.33_3.csv data_test_enc_0.33_3.csv data_test_enc_0.33_3.csv data_test_enc_r0.06_3.csv data_test_enc_r0.06_3.csv data_test_enc_r0.06_3.csv data_test_enc_r0.06_3.csv data_test_enc_r0.06_3.csv data_test_enc_r0.11_3.csv data_test_enc_r0.11_3.csv data_test_enc_r0.11_3.csv data_test_enc_r0.11_3.csv data_test_enc_r0.11_3.csv data_test_enc_0.33_4.csv data_test_enc_0.33_4.csv data_test_enc_0.33_4.csv data_test_enc_0.33_4.csv data_test_enc_0.33_4.csv data_test_enc_r0.06_4.csv data_test_enc_r0.06_4.csv data_test_enc_r0.06_4.csv data_test_enc_r0.06_4.csv data_test_enc_r0.06_4.csv data_test_enc_r0.11_4.csv data_test_enc_r0.11_4.csv data_test_enc_r0.11_4.csv data_test_enc_r0.11_4.csv data_test_enc_r0.11_4.csv data_test_enc_0.33_5.csv data_test_enc_0.33_5.csv data_test_enc_0.33_5.csv data_test_enc_0.33_5.csv data_test_enc_0.33_5.csv data_test_enc_r0.06_5.csv data_test_enc_r0.06_5.csv data_test_enc_r0.06_5.csv data_test_enc_r0.06_5.csv data_test_enc_r0.06_5.csv data_test_enc_r0.11_5.csv data_test_enc_r0.11_5.csv data_test_enc_r0.11_5.csv data_test_enc_r0.11_5.csv data_test_enc_r0.11_5.csv
"
depth_list="2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
"
pred_type_list="tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree
"
data_group_list="Warfarin_seed1 Warfarin_seed2 Warfarin_seed3 Warfarin_seed4 Warfarin_seed5 Warfarin_seed1 Warfarin_seed2 Warfarin_seed3 Warfarin_seed4 Warfarin_seed5 Warfarin_seed1 Warfarin_seed2 Warfarin_seed3 Warfarin_seed4 Warfarin_seed5 Warfarin_seed1 Warfarin_seed2 Warfarin_seed3 Warfarin_seed4 Warfarin_seed5 Warfarin_seed1 Warfarin_seed2 Warfarin_seed3 Warfarin_seed4 Warfarin_seed5 Warfarin_seed1 Warfarin_seed2 Warfarin_seed3 Warfarin_seed4 Warfarin_seed5 Warfarin_seed1 Warfarin_seed2 Warfarin_seed3 Warfarin_seed4 Warfarin_seed5 Warfarin_seed1 Warfarin_seed2 Warfarin_seed3 Warfarin_seed4 Warfarin_seed5 Warfarin_seed1 Warfarin_seed2 Warfarin_seed3 Warfarin_seed4 Warfarin_seed5 Warfarin_seed1 Warfarin_seed2 Warfarin_seed3 Warfarin_seed4 Warfarin_seed5 Warfarin_seed1 Warfarin_seed2 Warfarin_seed3 Warfarin_seed4 Warfarin_seed5 Warfarin_seed1 Warfarin_seed2 Warfarin_seed3 Warfarin_seed4 Warfarin_seed5 Warfarin_seed1 Warfarin_seed2 Warfarin_seed3 Warfarin_seed4 Warfarin_seed5 Warfarin_seed1 Warfarin_seed2 Warfarin_seed3 Warfarin_seed4 Warfarin_seed5 Warfarin_seed1 Warfarin_seed2 Warfarin_seed3 Warfarin_seed4 Warfarin_seed5
"
fairness_bounds_list="1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
"
protected_cols_list="White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White White
"
train_data_list=($train_data_list)
test_data_list=($test_data_list)
depth_list=($depth_list)
pred_type_list=($pred_type_list)
data_group_list=($data_group_list)
fairness_bounds_list=($fairness_bounds_list)
protected_cols_list=($protected_cols_list)


python main_warfarin.py -f ${train_data_list[$SLURM_ARRAY_TASK_ID]} -e ${test_data_list[$SLURM_ARRAY_TASK_ID]} -d ${depth_list[$SLURM_ARRAY_TASK_ID]} -t 3600 -b 100 -p ${pred_type_list[$SLURM_ARRAY_TASK_ID]} -g ${data_group_list[$SLURM_ARRAY_TASK_ID]} -u ${fairness_bounds_list[$SLURM_ARRAY_TASK_ID]} -c ${protected_cols_list[$SLURM_ARRAY_TASK_ID]}
