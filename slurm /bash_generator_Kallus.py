import os
import sys

path = '/Users/sina/Documents/GitHub/prescriptive-trees/'
approach_name = 'Kallus' #

samples = [1,2,3,4,5]

time_limit = 3600



def put_qmark(s):
        s = "\""+s+"\""
        return s


def generate(data_group, depths, thresholds, array):
        global time_limit, samples
        slurm_file = 'slurm_'+approach_name +'_' + data_group + '_'+ str(time_limit)+".sh"
        dir="/scratch2/saghaei/prescriptive-trees/"+approach_name+"/"

        train_data_list=[]
        test_data_list=[]
        depth_list=[]
        for s in samples:
            for d in depths:
                for threshold in thresholds:
                    training_file = 'data_train_' + str(threshold) + '_' + str(s) + '.csv'
                    test_file = 'data_test_' + str(threshold) + '_' + str(s) + '.csv'
                    train_data_list.append(training_file)
                    test_data_list.append(test_file)
                    depth_list.append(d)


        S="#!/bin/bash\n"
        # S+="#SBATCH --ntasks=100\n"
        S+="#SBATCH --ntasks=1\n"
        S+="#SBATCH --cpus-per-task=4\n"
        S+="#SBATCH --mem-per-cpu=4GB\n"
        S+="#SBATCH --time=02:00:00\n"
        S+="#SBATCH --export=NONE\n"
        S+="#SBATCH --constraint=\"xeon-2640v4\"\n"
        S+="#SBATCH --array=0-"
        S+=str(array)
        S+="\n\n"
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
        S+='train_data_list=($train_data_list)'+ "\n"
        S+='test_data_list=($test_data_list)'+ "\n"
        S+='depth_list=($depth_list)'+ "\n"

        S+="\n"
        S+="\n"
        command = 'python main.py ' + '-f ' +'${train_data_list[$SLURM_ARRAY_TASK_ID]}' + ' -e ' +'${test_data_list[$SLURM_ARRAY_TASK_ID]}' + " -d " + '${depth_list[$SLURM_ARRAY_TASK_ID]}' + " -t " + str(time_limit) + " -b " + str(100)+ " -r kallus" + " -n 0 " + " -g " + data_group
        S+=command
        S+="\n"



        dest_dir=path
        f= open(dest_dir+slurm_file,"w+")
        f.write(S)
        f.close()
        print(slurm_file)



def main():
    data_group = 'Athey_v2_4000' #Warfarin_3000 Athey_v1_500  Athey_v2_4000
    depths = [1, 2]
    thresholds = [0.1, 0.25, 0.5, 0.75, 0.9]  #[0.1, 0.25, 0.5, 0.75, 0.9] [0.1, 0.33, 0.6, 0.85]
    array = (len(samples) * len(depths) * len(thresholds)) - 1
    generate(data_group, depths, thresholds, array)

    data_group = 'Athey_v1_500' #Warfarin_3000 Athey_v1_500  Athey_v2_4000
    depths = [1, 2, 3, 4]
    thresholds = [0.1, 0.25, 0.5, 0.75, 0.9]  #[0.1, 0.25, 0.5, 0.75, 0.9] [0.1, 0.33, 0.6, 0.85]
    array = (len(samples) * len(depths) * len(thresholds)) - 1
    generate(data_group, depths, thresholds, array)

    data_group = 'Warfarin_3000' #Warfarin_3000 Athey_v1_500  Athey_v2_4000
    depths = [1, 2, 3, 4]
    thresholds = [0.1, 0.33, 0.6, 0.85]
    array = (len(samples) * len(depths) * len(thresholds)) - 1
    generate(data_group, depths, thresholds, array)


if __name__ == "__main__":
    main()
