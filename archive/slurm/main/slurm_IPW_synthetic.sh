#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --constraint="xeon-2640v4"
#SBATCH --array=0-49

cd /project/vayanou_651/prescriptive-trees/IPW/


module load gcc
module load gurobi
module load python
export PYTHONPATH=/project/vayanou_651/python/pkgs:${PYTHONPATH}


train_data_list="data_train_enc_0.1_1.csv data_train_enc_0.1_1.csv data_train_enc_0.25_1.csv data_train_enc_0.25_1.csv data_train_enc_0.5_1.csv data_train_enc_0.5_1.csv data_train_enc_0.75_1.csv data_train_enc_0.75_1.csv data_train_enc_0.9_1.csv data_train_enc_0.9_1.csv data_train_enc_0.1_2.csv data_train_enc_0.1_2.csv data_train_enc_0.25_2.csv data_train_enc_0.25_2.csv data_train_enc_0.5_2.csv data_train_enc_0.5_2.csv data_train_enc_0.75_2.csv data_train_enc_0.75_2.csv data_train_enc_0.9_2.csv data_train_enc_0.9_2.csv data_train_enc_0.1_3.csv data_train_enc_0.1_3.csv data_train_enc_0.25_3.csv data_train_enc_0.25_3.csv data_train_enc_0.5_3.csv data_train_enc_0.5_3.csv data_train_enc_0.75_3.csv data_train_enc_0.75_3.csv data_train_enc_0.9_3.csv data_train_enc_0.9_3.csv data_train_enc_0.1_4.csv data_train_enc_0.1_4.csv data_train_enc_0.25_4.csv data_train_enc_0.25_4.csv data_train_enc_0.5_4.csv data_train_enc_0.5_4.csv data_train_enc_0.75_4.csv data_train_enc_0.75_4.csv data_train_enc_0.9_4.csv data_train_enc_0.9_4.csv data_train_enc_0.1_5.csv data_train_enc_0.1_5.csv data_train_enc_0.25_5.csv data_train_enc_0.25_5.csv data_train_enc_0.5_5.csv data_train_enc_0.5_5.csv data_train_enc_0.75_5.csv data_train_enc_0.75_5.csv data_train_enc_0.9_5.csv data_train_enc_0.9_5.csv
"
test_data_list="data_test_enc_0.1_1.csv data_test_enc_0.1_1.csv data_test_enc_0.25_1.csv data_test_enc_0.25_1.csv data_test_enc_0.5_1.csv data_test_enc_0.5_1.csv data_test_enc_0.75_1.csv data_test_enc_0.75_1.csv data_test_enc_0.9_1.csv data_test_enc_0.9_1.csv data_test_enc_0.1_2.csv data_test_enc_0.1_2.csv data_test_enc_0.25_2.csv data_test_enc_0.25_2.csv data_test_enc_0.5_2.csv data_test_enc_0.5_2.csv data_test_enc_0.75_2.csv data_test_enc_0.75_2.csv data_test_enc_0.9_2.csv data_test_enc_0.9_2.csv data_test_enc_0.1_3.csv data_test_enc_0.1_3.csv data_test_enc_0.25_3.csv data_test_enc_0.25_3.csv data_test_enc_0.5_3.csv data_test_enc_0.5_3.csv data_test_enc_0.75_3.csv data_test_enc_0.75_3.csv data_test_enc_0.9_3.csv data_test_enc_0.9_3.csv data_test_enc_0.1_4.csv data_test_enc_0.1_4.csv data_test_enc_0.25_4.csv data_test_enc_0.25_4.csv data_test_enc_0.5_4.csv data_test_enc_0.5_4.csv data_test_enc_0.75_4.csv data_test_enc_0.75_4.csv data_test_enc_0.9_4.csv data_test_enc_0.9_4.csv data_test_enc_0.1_5.csv data_test_enc_0.1_5.csv data_test_enc_0.25_5.csv data_test_enc_0.25_5.csv data_test_enc_0.5_5.csv data_test_enc_0.5_5.csv data_test_enc_0.75_5.csv data_test_enc_0.75_5.csv data_test_enc_0.9_5.csv data_test_enc_0.9_5.csv
"
depth_list="1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
"
pred_type_list="tree log tree log tree log tree log tree log tree log tree log tree log tree log tree log tree log tree log tree log tree log tree log tree log tree log tree log tree log tree log tree log tree log tree log tree log tree log
"
data_group_list="Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500
"
train_data_list=($train_data_list)
test_data_list=($test_data_list)
depth_list=($depth_list)
pred_type_list=($pred_type_list)
data_group_list=($data_group_list)


python main.py -f ${train_data_list[$SLURM_ARRAY_TASK_ID]} -e ${test_data_list[$SLURM_ARRAY_TASK_ID]} -d ${depth_list[$SLURM_ARRAY_TASK_ID]} -t 3600 -b 100 -p ${pred_type_list[$SLURM_ARRAY_TASK_ID]} -g ${data_group_list[$SLURM_ARRAY_TASK_ID]}
