from gurobipy import *
import pandas as pd
import sys
import time
import Tree
import Primal_LAHSA
import logger
import getopt
import csv
import os
import pickle

##########################################################
# Functions
##########################################################

def get_node_status(primal, tree, b, w, p, n):
    pruned = False
    branching = False
    leaf = False
    treatment = None
    selected_feature = None

    p_sum = 0
    for m in tree.get_ancestors(n):
        p_sum = p_sum + p[m]
    if p[n] > 0.5:  # leaf
        leaf = True
        for k in primal.treatments_set:
            if w[n, k] > 0.5:
                treatment = k
    elif p_sum == 1:  # Pruned
        pruned = True

    if n in tree.Nodes:
        if pruned == False and leaf == False:  # branching
            for f in primal.features:
                if b[n, f] > 0.5:
                    selected_feature = f
                    branching = True

    return pruned, branching, selected_feature, leaf, treatment


def print_tree(primal, tree, b, w, p):
    for n in tree.Nodes + tree.Terminals:
        pruned, branching, selected_feature, leaf, treatment = get_node_status(primal, tree, b, w, p, n)
        print('#########node ', n)
        if pruned:
            print("pruned")
        elif branching:
            print(selected_feature)
        elif leaf:
            print('leaf {}'.format(treatment))


def print_tree_write(primal, tree, b, w, p, f):
    for n in tree.Nodes + tree.Terminals:
        pruned, branching, selected_feature, leaf, treatment = get_node_status(primal, tree, b, w, p, n)
        f.write(f'#########node  {n}\n')
        if pruned:
            f.write("pruned\n")
        elif branching:
            f.write(f'{selected_feature}\n')
        elif leaf:
            f.write(f'leaf {treatment}\n')


def get_predicted_value(primal, tree, b, w, p, local_data, i):
    current = 1

    while True:
        pruned, branching, selected_feature, leaf, treatment = get_node_status(primal, tree, b, w, p, current)
        if leaf:
            return treatment
            break
        elif branching:
            if local_data.at[i, selected_feature] == 1:  # going right on the branch
                current = tree.get_right_children(current)
            else:  # going left on the branch
                current = tree.get_left_children(current)


def get_metrics(primal, tree, b, w, p, local_data, true_outcome_cols, treatment_col):
    regret = 0
    best_found = 0
    treatment_classification_acc = 0
    for i in local_data.index:
        treatment_i_pred = get_predicted_value(primal, tree, b, w, p, local_data, i)
        received_treatment = local_data.at[i, treatment_col]
        if treatment_i_pred == received_treatment:
            treatment_classification_acc += 1
        pred_outcome = local_data.at[i, true_outcome_cols[treatment_i_pred]]
        best_outcome = 0
        for t in primal.treatments_set:
            if local_data.at[i, true_outcome_cols[t]] > best_outcome:
                best_outcome = local_data.at[i, true_outcome_cols[t]]
                best_treatment = t

        regret_i = best_outcome - pred_outcome
        regret += regret_i
        if regret_i == 0:
            best_found += 1

    return regret, (best_found / len(local_data) * 100), (treatment_classification_acc / len(local_data) * 100)


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "f:d:t:r:o:",
                                   ["training_file=", "depth=", "time_limit=",
                                    "method=", "out_file="])
    except getopt.GetoptError:
        sys.exit(2)

    training_file = None
    depth = None
    time_limit = None
    method = None
    out_file = None


    for opt, arg in opts:
        if opt in ("-f", "--training_file"):
            training_file = arg
        elif opt in ("-d", "--depth"):
            depth = int(arg)
        elif opt in ("-t", "--time_limit"):
            time_limit = int(arg)
        elif opt in ("-r", "--method"):
            method = arg
        elif opt in ("-o", "--out_file"):
            out_file = arg

    if not os.path.exists(out_file):
        os.makedirs(out_file)

    treatment_col = 't'
    outcome = 'y'
    prob_t = 'ipw'
    ########################
    # New variables defined for disability
    disab_col = 'DisablingCondition'
    pshTr = [2,3]
    ########################

    data = pd.read_csv(training_file)# file name
    features = [item for item in data.columns.tolist() if item not in ['t','ipw','y','y0', 'y1', 'y2', 'y3']] # feature we want to include
    # features = ['EnrollAge1',
    #  'EnrollAge2',
    #  'EnrollAge3',
    #  'EnrollAge4',
    #  'EnrollAge5',
    #  'Race_BlackAfAmerican',
    #  'Race_Hispanic',
    #  'Race_White',
    #  'Race_Other',
    #  'Gender_Female',
    #  'Gender_Male',
    #  'Gender_Trans',
    #  'PriorLivingCategory_Homeless Situations',
    #  'PriorLivingCategory_Institutional Situations',
    #  'PriorLivingCategory_Temporary and Permanent Housing Situations',
    #  'SleepLoc_Other',
    #  'SleepLoc_Outdoors',
    #  'SleepLoc_Safe Haven',
    #  'SleepLoc_Shelters',
    #  'SleepLoc_Transitional Housing',
    #  'score1',
    #  'score2',
    #  'score3',
    #  'score4',
    #  'VeteranStatus_0',
    #  'VeteranStatus_1',
    #  'TotalNumPrior1',
    #  'TotalNumPrior2',
    #  'TotalNumPrior3'] # feature we want to include

    y_hat = ['y0', 'y1', 'y2', 'y3']

    b_warm = None
    z_greedy = None
    warm_start_depth = 1

    constraints = {}
    for i in range(1, 4):
        constraints[i] = len(data[data[treatment_col] == i])/float(len(data))

    # CHECK IF EXISTING MST and PICKLE FILE
    if len(os.listdir(out_file)) > 0:
        list_of_solved = {}
        for i in os.listdir(out_file):
            if 'txt' not in i:
                file_type = i.split('.')[1]
                d = int(i.split('.')[0].split('_')[1])
                if d not in list_of_solved:
                    list_of_solved[d] = 1
                else:
                    list_of_solved[d] += 1

        # take the max key
        max_depth_done = max(list_of_solved.keys())
        num_solved = list_of_solved[max_depth_done]

        # this check might be overkill, but just wanted safety
        for i in range(1, max_depth_done):
            assert list_of_solved[i] == 3

        if num_solved == 1:
            # then the system stopped at warm start for that depth. proceed to solve the full solution for that depth using .mst
            if max_depth_done > 1:
                assert os.path.exists(os.path.join(out_file, f'branching_{max_depth_done-1}.pkl'))

                with open(os.path.join(out_file, f'branching_{max_depth_done-1}.pkl'), 'rb') as f:
                    b_warm = pickle.load(f)

            tree = Tree.Tree(max_depth_done)

            # ------ LOAD IN WARM START AND SOLVE ON ENTIRE DATASET -----
            primal = Primal_LAHSA.Primal(data, features, treatment_col, disab_col, pshTr, outcome, prob_t, tree, 
                time_limit, method, y_hat, b_warm, constraints, z_greedy)

            start_time = time.time()
            primal.create_primal_problem()

            primal.model.update()

            assert os.path.exists(os.path.join(out_file, f'warm_{max_depth_done}.mst'))
            primal.model.read(os.path.join(out_file, f'warm_{max_depth_done}.mst'))

            primal.model.optimize()
            end_time = time.time()
            solving_time_complete = end_time - start_time


            b_warm = primal.model.getAttr('X', primal.b)
            z_greedy = primal.model.getAttr('X', primal.z)
            with open(os.path.join(out_file, f'branching_{max_depth_done}.pkl'), 'wb') as f:
                pickle.dump(dict(b_warm), f)

            with open(os.path.join(out_file, f'z_{max_depth_done}.pkl'), 'wb') as f:
                pickle.dump(dict(z_greedy), f)

        elif num_solved == 3: # just load in the last branch
            assert os.path.exists(os.path.join(out_file, f'branching_{max_depth_done}.pkl'))
            with open(os.path.join(out_file, f'branching_{max_depth_done}.pkl'), 'rb') as f:
                b_warm = pickle.load(f)

            assert os.path.exists(os.path.join(out_file, f'z_{max_depth_done}.pkl'))
            with open(os.path.join(out_file, f'z_{max_depth_done}.pkl'), 'rb') as f:
                z_greedy = pickle.load(f)

        else:
            print(f'Only two files were found for depth {max_depth_done} -- this likely means the code stopped before pickling the z variables')
            return 
        
        warm_start_depth = max_depth_done + 1


    for warm_depth in range(warm_start_depth, depth+1):
        tree = Tree.Tree(warm_depth)  # Tree structure: We create a complete binary tree of depth d

        # solve warm start with a subset (5000) of the data first
        primal = Primal_LAHSA.Primal(data.sample(5000), features, treatment_col, disab_col, pshTr, 
            outcome, prob_t, tree, time_limit, method, y_hat, b_warm, constraints, z_greedy)

        start_time1 = time.time()
        primal.create_primal_problem()

        primal.model.update()
        primal.model.optimize()
        primal.model.write(os.path.join(out_file, f'warm_{warm_depth}.mst'))

        end_time1 = time.time()
        solving_time_warm = end_time1 - start_time1

        print(f'---- DEPTH {warm_start_depth} -----')
        print(f'\nWarm Start Solving Time: {solving_time_warm}')

        with open(os.path.join(out_file, 'log.txt'), 'a') as f:
            f.write(f'---- DEPTH {warm_start_depth} -----')
            f.write(f'\nWarm Start Solving Time: {solving_time_warm}')


        # ------ LOAD IN WARM START AND SOLVE ON ENTIRE DATASET -----
        primal = Primal_LAHSA.Primal(data, features, treatment_col, disab_col, pshTr, outcome, prob_t, tree, 
            time_limit, method, y_hat, b_warm, constraints, z_greedy)

        start_time = time.time()
        primal.create_primal_problem()

        primal.model.update()

        primal.model.read(os.path.join(out_file, f'warm_{warm_depth}.mst'))

        primal.model.optimize()
        end_time = time.time()
        solving_time_complete = end_time - start_time


        b_warm = primal.model.getAttr('X', primal.b)
        z_greedy = primal.model.getAttr('X', primal.z)

        with open(os.path.join(out_file, f'branching_{warm_depth}.pkl'), 'wb') as f:
            pickle.dump(dict(b_warm), f)

        with open(os.path.join(out_file, f'z_{warm_depth}.pkl'), 'wb') as f:
            pickle.dump(dict(z_greedy), f)

        print(f'\nFull Dataset Solving Time: {solving_time_complete}')
        print_tree(primal, tree,
                   primal.model.getAttr("X", primal.b),
                   primal.model.getAttr("X", primal.w),
                   primal.model.getAttr("X", primal.p))
        print(f"obj value: {primal.model.getAttr('ObjVal')}")
        print('\n\n')


        with open(os.path.join(out_file, 'log.txt'), 'a') as f:
            f.write(f'\nFull Dataset Solving Time: {solving_time_complete}')
            print_tree_write(primal, tree,
                   primal.model.getAttr("X", primal.b),
                   primal.model.getAttr("X", primal.w),
                   primal.model.getAttr("X", primal.p), f)
            f.write(f"obj value: {primal.model.getAttr('ObjVal')}")
            f.write('\n\n')

if __name__ == "__main__":
    main(sys.argv[1:])