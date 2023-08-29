import os
import sys

path = './'
approach_name = 'Direct_Approach' #

samples = [1, 2, 3, 4, 5]

time_limit = 10800



def put_qmark(s):
        s = "\""+s+"\""
        return s


def generate(data_groups, depths, thresholds, preds, mls, protected_cols, fairness_bounds, treatment_options, array, slurm_file):
        global time_limit, samples
        dir="/project/vayanou_651/prescriptive-trees/"+approach_name+"/"

        train_data_list=[]
        test_data_list=[]
        depth_list=[]
        pred_type_list=[]
        ml_list = []
        data_group_list = []
        fairness_bounds_list = []
        protected_cols_list = []
        treatment_options_list = []
        for s in samples:
            for d in depths:
                for threshold in thresholds:
                    for pred in preds:
                        for ml in mls:
                            for group in data_groups:
                                for f in fairness_bounds:
                                    for c in protected_cols:
                                        training_file = 'data_train_enc_' + str(threshold) + '_' + str(s) + '.csv'
                                        test_file = 'data_test_enc_' + str(threshold) + '_' + str(s) + '.csv'
                                        train_data_list.append(training_file)
                                        test_data_list.append(test_file)
                                        depth_list.append(d)
                                        pred_type_list.append(pred)
                                        ml_list.append(ml)
                                        data_group_list.append(group)
                                        fairness_bounds_list.append(f)
                                        protected_cols_list.append(c)
                                        treatment_options_list.append(treatment_options)


        S="#!/bin/bash\n"
        # S+="#SBATCH --ntasks=100\n"
        S+="#SBATCH --ntasks=1\n"
        S+="#SBATCH --cpus-per-task=4\n"
        S+="#SBATCH --mem-per-cpu=6GB\n"
        S+="#SBATCH --time=03:30:00\n"
        S+="#SBATCH --export=NONE\n"
        S+="#SBATCH --constraint=\"xeon-2640v4\"\n"
        S+="#SBATCH --mail-type=ALL\n"
        S+="#SBATCH --mail-user=nathanael.jo@gmail.com\n"
        S+="#SBATCH --array=0-"
        S+=str(array)
        S+="\n\n"
        S+="cd "+dir+"\n"
        S+="\n"
        S+="\n"

        S+="module load gcc/11.3.0\n"
        S+="module load gurobi/10.0.0\n"
        S+="module load python/3.9.12\n"
        S+="export PYTHONPATH=/project/vayanou_651/python/pkgs:${PYTHONPATH}\n"
        S+="\n"
        S+="\n"

        S+="train_data_list=" + put_qmark(" ".join(str(item) for item in train_data_list) + "\n")
        S+="\n"
        S+="test_data_list=" + put_qmark(" ".join(str(item) for item in test_data_list) + "\n")
        S+="\n"
        S+="depth_list=" + put_qmark(" ".join(str(item) for item in depth_list) + "\n")
        S+="\n"
        S+="pred_type_list=" + put_qmark(" ".join(str(item) for item in pred_type_list) + "\n")
        S+="\n"
        S+="data_group_list=" + put_qmark(" ".join(str(item) for item in data_group_list) + "\n")
        S+="\n"
        S+="ml_list=" + put_qmark(" ".join(str(item) for item in ml_list) + "\n")
        S+="\n"
        S+="fairness_bounds_list=" + put_qmark(" ".join(str(item) for item in fairness_bounds_list) + "\n")
        S+="\n"
        S+="protected_cols_list=" + put_qmark(" ".join(str(item) for item in protected_cols_list) + "\n")
        S+="\n"
        S+="treatment_options_list=" + put_qmark(" ".join(str(item) for item in treatment_options_list) + "\n")
        S+="\n"
        S+='train_data_list=($train_data_list)'+ "\n"
        S+='test_data_list=($test_data_list)'+ "\n"
        S+='depth_list=($depth_list)'+ "\n"
        S+='pred_type_list=($pred_type_list)'+ "\n"
        S+='data_group_list=($data_group_list)'+ "\n"
        S+='ml_list=($ml_list)'+ "\n"
        S+='fairness_bounds_list=($fairness_bounds_list)'+ "\n"
        S+='protected_cols_list=($protected_cols_list)'+ "\n"
        S+='treatment_options_list=($treatment_options_list)'+ "\n"


        S+="\n"
        S+="\n"
        command = 'python main_agg.py ' + '-f ' +'${train_data_list[$SLURM_ARRAY_TASK_ID]}' + ' -e ' +'${test_data_list[$SLURM_ARRAY_TASK_ID]}' + " -d " + '${depth_list[$SLURM_ARRAY_TASK_ID]}' + " -t " + str(time_limit) + " -b " + str(100)+ " -p " + '${pred_type_list[$SLURM_ARRAY_TASK_ID]}'+ " -r robust" + " -g " + '${data_group_list[$SLURM_ARRAY_TASK_ID]}' + " -m " + '${ml_list[$SLURM_ARRAY_TASK_ID]}' + " -u " + '${treatment_options_list[$SLURM_ARRAY_TASK_ID]}' + " -a " + '${fairness_bounds_list[$SLURM_ARRAY_TASK_ID]}' + " -c " + '${protected_cols_list[$SLURM_ARRAY_TASK_ID]}'
        S+=command
        S+="\n"



        dest_dir=path
        f= open(dest_dir+slurm_file,"w+")
        f.write(S)
        f.close()
        print(slurm_file)



def main():
    # data_group = 'Athey_v2_4000' #Warfarin_3000 Athey_v1_500  Athey_v2_4000
    # depths = [1, 2]
    # thresholds = [0.1, 0.25, 0.5, 0.75, 0.9]  #[0.1, 0.25, 0.5, 0.75, 0.9] [0.1, 0.33, 0.6, 0.85]
    # preds = ['true', 'tree', 'log']
    # mls = ['linear', 'lasso']
    # array = (len(samples) * len(depths) * len(thresholds) * len(preds) * len(mls)) - 1
    # generate(data_group, depths, thresholds, preds, mls, array)

    # data_group = ['Athey_v1_500'] #Warfarin_3000 Athey_v1_500  Athey_v2_4000
    # depths = [1, 2]
    # budgets = [0.2]
    # thresholds = [0.1, 0.25, 0.5, 0.75, 0.9]  #[0.1, 0.25, 0.5, 0.75, 0.9] [0.1, 0.33, 0.6, 0.85]
    # preds = ['true', 'tree', 'log']  #['true', 'tree', 'log'] ['tree']
    # mls = ['linear', 'lasso']
    # array = (len(samples) * len(depths) * len(thresholds) * len(preds) * len(mls) * len(budgets)) - 1
    # generate(data_group, depths, thresholds, preds, mls, budgets, array)

    # data_groups = ['Warfarin_seed1','Warfarin_seed2','Warfarin_seed3','Warfarin_seed4','Warfarin_seed5'] #Warfarin_3000 Athey_v1_500  Athey_v2_4000
    # # data_groups = ['Warfarin_seed4'] 
    # slurm_file = 'slurm_DR_' + 'Warfarinv2.sh'
    # depths = [2]
    # thresholds = ["r0.06", "r0.11"]
    # preds = ['tree']  #['true', 'tree', 'log'] ['tree']
    # mls = ['lrrf']
    # protected_cols = ['White']
    # fairness_bounds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    # treatment_options = 3
    # array = (len(samples) * len(depths) * len(thresholds) * len(preds) * len(mls) * len(data_groups) * len(protected_cols) * len(fairness_bounds)) - 1
    # generate(data_groups, depths, thresholds, preds, mls, protected_cols, fairness_bounds, treatment_options, array, slurm_file)

    # data_groups = ['Warfarin_seed1','Warfarin_seed2','Warfarin_seed3','Warfarin_seed4','Warfarin_seed5'] #Warfarin_3000 Athey_v1_500  Athey_v2_4000
    # # data_groups = ['Warfarin_seed4'] 
    # slurm_file = 'slurm_DR_' + 'Warfarinv2_0.33.sh'
    # depths = [2]
    # thresholds = ["0.33"]
    # preds = ['tree']  #['true', 'tree', 'log'] ['tree']
    # mls = ['ml']
    # protected_cols = ['White']
    # fairness_bounds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    # treatment_options = 3
    # array = (len(samples) * len(depths) * len(thresholds) * len(preds) * len(mls) * len(data_groups) * len(protected_cols) * len(fairness_bounds)) - 1
    # generate(data_groups, depths, thresholds, preds, mls, protected_cols, fairness_bounds, treatment_options, array, slurm_file)


    data_groups = ['Warfarin_seed1'] #Warfarin_3000 Athey_v1_500  Athey_v2_4000
    # data_groups = ['Warfarin_seed4'] 
    slurm_file = 'slurm_DR_' + 'Warfarinv2_test.sh'
    depths = [2]
    thresholds = ["0.33"]
    preds = ['tree']  #['true', 'tree', 'log'] ['tree']
    mls = ['ml']
    protected_cols = ['White']
    fairness_bounds = [0.08]
    treatment_options = 3
    array = (len(samples) * len(depths) * len(thresholds) * len(preds) * len(mls) * len(data_groups) * len(protected_cols) * len(fairness_bounds)) - 1
    generate(data_groups, depths, thresholds, preds, mls, protected_cols, fairness_bounds, treatment_options, array, slurm_file)



if __name__ == "__main__":
    main()
