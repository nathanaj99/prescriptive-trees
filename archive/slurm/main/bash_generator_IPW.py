import os
import sys

path = '/Users/sina/Documents/GitHub/prescriptive-trees/'
approach_name = 'IPW' #
data_group = 'Athey_v2_4000' #Warfarin_3000 Athey_v1_500  Athey_v2_4000
samples = [1,2,3,4,5]
depths = [1, 2]
thresholds = [0.1, 0.25, 0.5, 0.75, 0.9]  #[0.1, 0.25, 0.5, 0.75, 0.9] [0.1, 0.33, 0.6, 0.85]
preds = ['true', 'tree', 'log']  #['true', 'tree', 'log'] ['tree']
time_limit = 3600



def put_qmark(s):
        s = "\""+s+"\""
        return s


def generate():
        global time_limit, depths, samples, thresholds, preds, approach_name, data_group
        slurm_file = 'slurm_'+approach_name +'_' + data_group + '_'+ str(time_limit)+".sh"
        dir="/scratch2/saghaei/prescriptive-trees/"+approach_name+"/"

        train_data_list=[]
        test_data_list=[]
        depth_list=[]
        pred_type_list=[]
        for s in samples:
            for d in depths:
                for threshold in thresholds:
                    for pred in preds:
                            training_file = 'data_train_' + str(threshold) + '_' + str(s) + '.csv'
                            test_file = 'data_test_' + str(threshold) + '_' + str(s) + '.csv'
                            train_data_list.append(training_file)
                            test_data_list.append(test_file)
                            depth_list.append(d)
                            pred_type_list.append(pred)

        S="#!/bin/bash\n"
        # S+="#SBATCH --ntasks=100\n"
        S+="#SBATCH --ntasks=1\n"
        S+="#SBATCH --cpus-per-task=4\n"
        S+="#SBATCH --mem-per-cpu=4GB\n"
        S+="#SBATCH --time=02:00:00\n"
        S+="#SBATCH --export=NONE\n"
        S+="#SBATCH --constraint=\"xeon-2640v4\"\n"
        S+="#SBATCH --array=0-149\n"
        S+="\n"
        S+="cd "+dir+"\n"
        S+="\n"
        S+="\n"
        S+="module load gcc\n"
        S+="module load gurobi\n"
        S+="module load python\n"
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
        S+='train_data_list=($train_data_list)'+ "\n"
        S+='test_data_list=($test_data_list)'+ "\n"
        S+='depth_list=($depth_list)'+ "\n"
        S+='pred_type_list=($pred_type_list)'+ "\n"

        S+="\n"
        S+="\n"
        command = 'python main.py ' + '-f ' +'${train_data_list[$SLURM_ARRAY_TASK_ID]}' + ' -e ' +'${test_data_list[$SLURM_ARRAY_TASK_ID]}' + " -d " + '${depth_list[$SLURM_ARRAY_TASK_ID]}' + " -t " + str(time_limit) + " -b " + str(100)+ " -p " + '${pred_type_list[$SLURM_ARRAY_TASK_ID]}'+ " -g " + data_group
        S+=command
        S+="\n"



        dest_dir=path
        f= open(dest_dir+slurm_file,"w+")
        f.write(S)
        f.close()
        print(slurm_file)










def main():
        generate()




if __name__ == "__main__":
    main()
