#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --constraint="xeon-2640v4"
#SBATCH --array=0-13

cd /scratch2/saghaei/prescriptive-trees/Kallus_Bertsimas/


module load gcc
module load gurobi
module load python


train_data_list="data_train_enc_0.1_4.csv data_train_enc_0.1_1.csv data_train_enc_0.1_3.csv data_train_enc_0.1_4.csv data_train_enc_0.33_2.csv data_train_enc_0.33_4.csv data_train_enc_0.6_1.csv data_train_enc_0.6_2.csv data_train_enc_0.6_3.csv data_train_enc_0.6_4.csv data_train_enc_0.6_5.csv data_train_enc_0.85_1.csv data_train_enc_0.85_4.csv data_train_0.75_5.csv
"
test_data_list="data_test_enc_0.1_4.csv data_test_enc_0.1_1.csv data_test_enc_0.1_3.csv data_test_enc_0.1_4.csv data_test_enc_0.33_2.csv data_test_enc_0.33_4.csv data_test_enc_0.6_1.csv data_test_enc_0.6_2.csv data_test_enc_0.6_3.csv data_test_enc_0.6_4.csv data_test_enc_0.6_5.csv data_test_enc_0.85_1.csv data_test_enc_0.85_4.csv data_test_0.75_5.csv
"
depth_list="2 1 1 1 1 1 1 1 1 1 1 1 1 1
"
type_list="kallus bertsimas bertsimas bertsimas bertsimas bertsimas bertsimas bertsimas bertsimas bertsimas bertsimas bertsimas bertsimas bertsimas
"
group_list="Warfarin_3000 Warfarin_3000 Warfarin_3000 Warfarin_3000 Warfarin_3000 Warfarin_3000 Warfarin_3000 Warfarin_3000 Warfarin_3000 Warfarin_3000 Warfarin_3000 Warfarin_3000 Warfarin_3000 Athey_v2_4000"
train_data_list=($train_data_list)
test_data_list=($test_data_list)
depth_list=($depth_list)
type_list=($type_list)
group_list=($group_list)


python main.py -f ${train_data_list[$SLURM_ARRAY_TASK_ID]} -e ${test_data_list[$SLURM_ARRAY_TASK_ID]} -d ${depth_list[$SLURM_ARRAY_TASK_ID]} -t 3600 -b 100 -r ${type_list[$SLURM_ARRAY_TASK_ID]} -n 0 -g ${group_list[$SLURM_ARRAY_TASK_ID]} 
