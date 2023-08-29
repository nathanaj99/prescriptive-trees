#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=01:10:00
#SBATCH --export=NONE
#SBATCH --constraint="xeon-2640v4"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nathanael.jo@gmail.com
#SBATCH --array=0-19

cd /project/vayanou_651/prescriptive-trees/Direct_Approach/


module load gcc/11.3.0
module load gurobi/10.0.0
module load python/3.9.12
export PYTHONPATH=/project/vayanou_651/python/pkgs:${PYTHONPATH}


train_data_list="data_train_enc_0.1_1.csv data_train_enc_0.1_1.csv data_train_enc_0.1_1.csv data_train_enc_0.1_1.csv data_train_enc_0.1_2.csv data_train_enc_0.1_2.csv data_train_enc_0.1_2.csv data_train_enc_0.1_2.csv data_train_enc_0.1_3.csv data_train_enc_0.1_3.csv data_train_enc_0.1_3.csv data_train_enc_0.1_3.csv data_train_enc_0.1_4.csv data_train_enc_0.1_4.csv data_train_enc_0.1_4.csv data_train_enc_0.1_4.csv data_train_enc_0.1_5.csv data_train_enc_0.1_5.csv data_train_enc_0.1_5.csv data_train_enc_0.1_5.csv
"
test_data_list="data_test_enc_0.1_1.csv data_test_enc_0.1_1.csv data_test_enc_0.1_1.csv data_test_enc_0.1_1.csv data_test_enc_0.1_2.csv data_test_enc_0.1_2.csv data_test_enc_0.1_2.csv data_test_enc_0.1_2.csv data_test_enc_0.1_3.csv data_test_enc_0.1_3.csv data_test_enc_0.1_3.csv data_test_enc_0.1_3.csv data_test_enc_0.1_4.csv data_test_enc_0.1_4.csv data_test_enc_0.1_4.csv data_test_enc_0.1_4.csv data_test_enc_0.1_5.csv data_test_enc_0.1_5.csv data_test_enc_0.1_5.csv data_test_enc_0.1_5.csv
"
depth_list="1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
"
pred_type_list="tree tree log log tree tree log log tree tree log log tree tree log log tree tree log log
"
data_group_list="Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500 Athey_v1_500
"
ml_list="linear lasso linear lasso linear lasso linear lasso linear lasso linear lasso linear lasso linear lasso linear lasso linear lasso
"
budgets_list="0.4 0.4 0.4 0.4 0.4 0.4 0.4 0.4 0.4 0.4 0.4 0.4 0.4 0.4 0.4 0.4 0.4 0.4 0.4 0.4
"
treatments_for_budget_list="1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
"
treatment_options_list="2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
"
train_data_list=($train_data_list)
test_data_list=($test_data_list)
depth_list=($depth_list)
pred_type_list=($pred_type_list)
data_group_list=($data_group_list)
ml_list=($ml_list)
budgets_list=($budgets_list)
treatments_for_budget_list=($treatments_for_budget_list)
treatment_options_list=($treatment_options_list)


python main_agg.py -f ${train_data_list[$SLURM_ARRAY_TASK_ID]} -e ${test_data_list[$SLURM_ARRAY_TASK_ID]} -d ${depth_list[$SLURM_ARRAY_TASK_ID]} -t 3600 -b 100 -p ${pred_type_list[$SLURM_ARRAY_TASK_ID]} -r robust -g ${data_group_list[$SLURM_ARRAY_TASK_ID]} -m ${ml_list[$SLURM_ARRAY_TASK_ID]} -u ${treatment_options_list[$SLURM_ARRAY_TASK_ID]} -n ${treatments_for_budget_list[$SLURM_ARRAY_TASK_ID]} -o ${budgets_list[$SLURM_ARRAY_TASK_ID]}
