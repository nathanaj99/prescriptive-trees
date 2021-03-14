#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --constraint="xeon-2640v4"
#SBATCH --array=0-79

cd /scratch/nathanaj/prescriptive-trees/IPW/


module load gcc
module load gurobi
module load python


train_data_list="data_train_enc_0.1_1.csv data_train_enc_0.33_1.csv data_train_enc_0.6_1.csv data_train_enc_0.85_1.csv data_train_enc_0.1_1.csv data_train_enc_0.33_1.csv data_train_enc_0.6_1.csv data_train_enc_0.85_1.csv data_train_enc_0.1_1.csv data_train_enc_0.33_1.csv data_train_enc_0.6_1.csv data_train_enc_0.85_1.csv data_train_enc_0.1_1.csv data_train_enc_0.33_1.csv data_train_enc_0.6_1.csv data_train_enc_0.85_1.csv data_train_enc_0.1_2.csv data_train_enc_0.33_2.csv data_train_enc_0.6_2.csv data_train_enc_0.85_2.csv data_train_enc_0.1_2.csv data_train_enc_0.33_2.csv data_train_enc_0.6_2.csv data_train_enc_0.85_2.csv data_train_enc_0.1_2.csv data_train_enc_0.33_2.csv data_train_enc_0.6_2.csv data_train_enc_0.85_2.csv data_train_enc_0.1_2.csv data_train_enc_0.33_2.csv data_train_enc_0.6_2.csv data_train_enc_0.85_2.csv data_train_enc_0.1_3.csv data_train_enc_0.33_3.csv data_train_enc_0.6_3.csv data_train_enc_0.85_3.csv data_train_enc_0.1_3.csv data_train_enc_0.33_3.csv data_train_enc_0.6_3.csv data_train_enc_0.85_3.csv data_train_enc_0.1_3.csv data_train_enc_0.33_3.csv data_train_enc_0.6_3.csv data_train_enc_0.85_3.csv data_train_enc_0.1_3.csv data_train_enc_0.33_3.csv data_train_enc_0.6_3.csv data_train_enc_0.85_3.csv data_train_enc_0.1_4.csv data_train_enc_0.33_4.csv data_train_enc_0.6_4.csv data_train_enc_0.85_4.csv data_train_enc_0.1_4.csv data_train_enc_0.33_4.csv data_train_enc_0.6_4.csv data_train_enc_0.85_4.csv data_train_enc_0.1_4.csv data_train_enc_0.33_4.csv data_train_enc_0.6_4.csv data_train_enc_0.85_4.csv data_train_enc_0.1_4.csv data_train_enc_0.33_4.csv data_train_enc_0.6_4.csv data_train_enc_0.85_4.csv data_train_enc_0.1_5.csv data_train_enc_0.33_5.csv data_train_enc_0.6_5.csv data_train_enc_0.85_5.csv data_train_enc_0.1_5.csv data_train_enc_0.33_5.csv data_train_enc_0.6_5.csv data_train_enc_0.85_5.csv data_train_enc_0.1_5.csv data_train_enc_0.33_5.csv data_train_enc_0.6_5.csv data_train_enc_0.85_5.csv data_train_enc_0.1_5.csv data_train_enc_0.33_5.csv data_train_enc_0.6_5.csv data_train_enc_0.85_5.csv
"
test_data_list="data_test_enc_0.1_1.csv data_test_enc_0.33_1.csv data_test_enc_0.6_1.csv data_test_enc_0.85_1.csv data_test_enc_0.1_1.csv data_test_enc_0.33_1.csv data_test_enc_0.6_1.csv data_test_enc_0.85_1.csv data_test_enc_0.1_1.csv data_test_enc_0.33_1.csv data_test_enc_0.6_1.csv data_test_enc_0.85_1.csv data_test_enc_0.1_1.csv data_test_enc_0.33_1.csv data_test_enc_0.6_1.csv data_test_enc_0.85_1.csv data_test_enc_0.1_2.csv data_test_enc_0.33_2.csv data_test_enc_0.6_2.csv data_test_enc_0.85_2.csv data_test_enc_0.1_2.csv data_test_enc_0.33_2.csv data_test_enc_0.6_2.csv data_test_enc_0.85_2.csv data_test_enc_0.1_2.csv data_test_enc_0.33_2.csv data_test_enc_0.6_2.csv data_test_enc_0.85_2.csv data_test_enc_0.1_2.csv data_test_enc_0.33_2.csv data_test_enc_0.6_2.csv data_test_enc_0.85_2.csv data_test_enc_0.1_3.csv data_test_enc_0.33_3.csv data_test_enc_0.6_3.csv data_test_enc_0.85_3.csv data_test_enc_0.1_3.csv data_test_enc_0.33_3.csv data_test_enc_0.6_3.csv data_test_enc_0.85_3.csv data_test_enc_0.1_3.csv data_test_enc_0.33_3.csv data_test_enc_0.6_3.csv data_test_enc_0.85_3.csv data_test_enc_0.1_3.csv data_test_enc_0.33_3.csv data_test_enc_0.6_3.csv data_test_enc_0.85_3.csv data_test_enc_0.1_4.csv data_test_enc_0.33_4.csv data_test_enc_0.6_4.csv data_test_enc_0.85_4.csv data_test_enc_0.1_4.csv data_test_enc_0.33_4.csv data_test_enc_0.6_4.csv data_test_enc_0.85_4.csv data_test_enc_0.1_4.csv data_test_enc_0.33_4.csv data_test_enc_0.6_4.csv data_test_enc_0.85_4.csv data_test_enc_0.1_4.csv data_test_enc_0.33_4.csv data_test_enc_0.6_4.csv data_test_enc_0.85_4.csv data_test_enc_0.1_5.csv data_test_enc_0.33_5.csv data_test_enc_0.6_5.csv data_test_enc_0.85_5.csv data_test_enc_0.1_5.csv data_test_enc_0.33_5.csv data_test_enc_0.6_5.csv data_test_enc_0.85_5.csv data_test_enc_0.1_5.csv data_test_enc_0.33_5.csv data_test_enc_0.6_5.csv data_test_enc_0.85_5.csv data_test_enc_0.1_5.csv data_test_enc_0.33_5.csv data_test_enc_0.6_5.csv data_test_enc_0.85_5.csv
"
depth_list="1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4
"
pred_type_list="tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree tree
"
train_data_list=($train_data_list)
test_data_list=($test_data_list)
depth_list=($depth_list)
pred_type_list=($pred_type_list)


python main.py -f ${train_data_list[$SLURM_ARRAY_TASK_ID]} -e ${test_data_list[$SLURM_ARRAY_TASK_ID]} -d ${depth_list[$SLURM_ARRAY_TASK_ID]} -t 3600 -b 100 -p ${pred_type_list[$SLURM_ARRAY_TASK_ID]} -g Warfarin_3000
