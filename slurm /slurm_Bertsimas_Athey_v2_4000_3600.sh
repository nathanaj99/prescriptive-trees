#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --constraint="xeon-2640v4"
#SBATCH --array=0-49

cd /scratch2/saghaei/prescriptive-trees/Kallus_Bertsimas/


module load gcc
module load gurobi
module load python


train_data_list="data_train_0.1_1.csv data_train_0.25_1.csv data_train_0.5_1.csv data_train_0.75_1.csv data_train_0.9_1.csv data_train_0.1_1.csv data_train_0.25_1.csv data_train_0.5_1.csv data_train_0.75_1.csv data_train_0.9_1.csv data_train_0.1_2.csv data_train_0.25_2.csv data_train_0.5_2.csv data_train_0.75_2.csv data_train_0.9_2.csv data_train_0.1_2.csv data_train_0.25_2.csv data_train_0.5_2.csv data_train_0.75_2.csv data_train_0.9_2.csv data_train_0.1_3.csv data_train_0.25_3.csv data_train_0.5_3.csv data_train_0.75_3.csv data_train_0.9_3.csv data_train_0.1_3.csv data_train_0.25_3.csv data_train_0.5_3.csv data_train_0.75_3.csv data_train_0.9_3.csv data_train_0.1_4.csv data_train_0.25_4.csv data_train_0.5_4.csv data_train_0.75_4.csv data_train_0.9_4.csv data_train_0.1_4.csv data_train_0.25_4.csv data_train_0.5_4.csv data_train_0.75_4.csv data_train_0.9_4.csv data_train_0.1_5.csv data_train_0.25_5.csv data_train_0.5_5.csv data_train_0.75_5.csv data_train_0.9_5.csv data_train_0.1_5.csv data_train_0.25_5.csv data_train_0.5_5.csv data_train_0.75_5.csv data_train_0.9_5.csv
"
test_data_list="data_test_0.1_1.csv data_test_0.25_1.csv data_test_0.5_1.csv data_test_0.75_1.csv data_test_0.9_1.csv data_test_0.1_1.csv data_test_0.25_1.csv data_test_0.5_1.csv data_test_0.75_1.csv data_test_0.9_1.csv data_test_0.1_2.csv data_test_0.25_2.csv data_test_0.5_2.csv data_test_0.75_2.csv data_test_0.9_2.csv data_test_0.1_2.csv data_test_0.25_2.csv data_test_0.5_2.csv data_test_0.75_2.csv data_test_0.9_2.csv data_test_0.1_3.csv data_test_0.25_3.csv data_test_0.5_3.csv data_test_0.75_3.csv data_test_0.9_3.csv data_test_0.1_3.csv data_test_0.25_3.csv data_test_0.5_3.csv data_test_0.75_3.csv data_test_0.9_3.csv data_test_0.1_4.csv data_test_0.25_4.csv data_test_0.5_4.csv data_test_0.75_4.csv data_test_0.9_4.csv data_test_0.1_4.csv data_test_0.25_4.csv data_test_0.5_4.csv data_test_0.75_4.csv data_test_0.9_4.csv data_test_0.1_5.csv data_test_0.25_5.csv data_test_0.5_5.csv data_test_0.75_5.csv data_test_0.9_5.csv data_test_0.1_5.csv data_test_0.25_5.csv data_test_0.5_5.csv data_test_0.75_5.csv data_test_0.9_5.csv
"
depth_list="1 1 1 1 1 2 2 2 2 2 1 1 1 1 1 2 2 2 2 2 1 1 1 1 1 2 2 2 2 2 1 1 1 1 1 2 2 2 2 2 1 1 1 1 1 2 2 2 2 2
"
train_data_list=($train_data_list)
test_data_list=($test_data_list)
depth_list=($depth_list)


python main.py -f ${train_data_list[$SLURM_ARRAY_TASK_ID]} -e ${test_data_list[$SLURM_ARRAY_TASK_ID]} -d ${depth_list[$SLURM_ARRAY_TASK_ID]} -t 3600 -b 100 -r bertsimas -n 0 -g Athey_v2_4000
