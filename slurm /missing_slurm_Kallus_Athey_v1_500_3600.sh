#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --constraint="xeon-2640v4"
#SBATCH --array=0-7

cd /scratch2/saghaei/prescriptive-trees/Kallus_Bertsimas/


module load gcc
module load gurobi
module load python


train_data_list="data_train_enc_0.1_1.csv data_train_enc_0.1_2.csv data_train_enc_0.5_3.csv data_train_enc_0.9_4.csv data_train_enc_0.1_5.csv data_train_enc_0.5_4.csv data_train_enc_0.9_2.csv data_train_enc_0.9_3.csv
"
test_data_list="data_test_enc_0.1_1.csv data_test_enc_0.1_2.csv data_test_enc_0.5_3.csv data_test_enc_0.9_4.csv data_test_enc_0.1_5.csv data_test_enc_0.5_4.csv data_test_enc_0.9_2.csv data_test_enc_0.9_3.csv
"
depth_list="2 2 2 3 3 3 3 2
"
type_list="kallus kallus kallus kallus bertsimas bertsimas bertsimas bertsimas
"
train_data_list=($train_data_list)
test_data_list=($test_data_list)
depth_list=($depth_list)
type_list = ($type_list)


python main.py -f ${train_data_list[$SLURM_ARRAY_TASK_ID]} -e ${test_data_list[$SLURM_ARRAY_TASK_ID]} -d ${depth_list[$SLURM_ARRAY_TASK_ID]} -t 3600 -b 100 -r ${type_list[$SLURM_ARRAY_TASK_ID]} -n 0 -g Athey_v1_500
