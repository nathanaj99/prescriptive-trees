#!/usr/bin/python

from gurobipy import *
import pandas as pd
import sys
import time
import Tree
import Primal
import logger
import getopt
import csv
import numpy as np


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
    print(argv)

    training_file = None
    test_file = None
    depth = None
    branching_limit = None
    time_limit = None
    prob_type_pred = None
    robust = None
    data_group = None
    ml = None

    try:
        opts, args = getopt.getopt(argv, "f:e:d:b:t:p:r:g:m:",
                                   ["training_file=", "test_file=", "depth=", "branching_limit=", "time_limit=",
                                    "pred=", "robust=", "data_group=", "ml="])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-f", "--training_file"):
            training_file = arg
        elif opt in ("-e", "--test_file"):
            test_file = arg
        elif opt in ("-d", "--depth"):
            depth = int(arg)
        elif opt in ("-b", "--branching_limit"):
            branching_limit = float(arg)
        elif opt in ("-t", "--time_limit"):
            time_limit = int(arg)
        elif opt in ("-p", "--pred"):
            prob_type_pred = arg
        elif opt in ("-r", "--robust"):
            robust = arg
        elif opt in ("-g", "--data_group"):
            data_group = arg
        elif opt in ("-m", "--ml"):
            ml = arg

    # ---- CHANGE FILE PATH ----
    if robust == 'robust':
        robust = True
    elif robust == 'direct':
        robust = False

    data_path_dict = {}
    if robust:
        data_path_dict = {'Warfarin_seed1': ('/../data/Warfarin_v2/seed1/', '/../Results_Robust/Warfarin_v2/seed1/'),
                      'Warfarin_seed2': ('/../data/Warfarin_v2/seed2/', '/../Results_Robust/Warfarin_v2/seed2/'),
                      'Warfarin_seed3': ('/../data/Warfarin_v2/seed3/', '/../Results_Robust/Warfarin_v2/seed3/'),
                      'Warfarin_seed4': ('/../data/Warfarin_v2/seed4/', '/../Results_Robust/Warfarin_v2/seed4/'),
                      'Warfarin_seed5': ('/../data/Warfarin_v2/seed5/', '/../Results_Robust/Warfarin_v2/seed5/'),
                          'Athey_v1_500': ('/../data/Athey_v1/500/', '/../Results_Robust/Athey_v1/500/'),
                          'Athey_v2_4000': ('/../data/Athey_v2/4000/', '/../Results_Robust/Athey_v2/4000/')}
    else:
        data_path_dict = {'Warfarin_seed1': ('/../data/Warfarin_v2/seed1/', '/../Results_Direct/Warfarin_v2/seed1/'),
                      'Warfarin_seed2': ('/../data/Warfarin_v2/seed2/', '/../Results_DirectWarfarin_v2/seed2/'),
                      'Warfarin_seed3': ('/../data/Warfarin_v2/seed3/', '/../Results_Direct/Warfarin_v2/seed3/'),
                      'Warfarin_seed4': ('/../data/Warfarin_v2/seed4/', '/../Results_Direct/Warfarin_v2/seed4/'),
                      'Warfarin_seed5': ('/../data/Warfarin_v2/seed5/', '/../Results_Direct/Warfarin_v2/seed5/'),
                          'Athey_v1_500': ('/../data/Athey_v1/500/', '/../Results_Direct/Athey_v1/500/'),
                          'Athey_v2_4000': ('/../data/Athey_v2/4000/', '/../Results_Direct/Athey_v2/4000/')}

    data_group_features_dict = {
        'Warfarin_3000': ['Age1.2', 'Age3.4', 'Age5.6', 'Age7', 'Age8.9', 'Height1', 'Height2', 'Height3', 'Height4',
                          'Height5',
                          'Weight1', 'Weight2', 'Weight3', 'Weight4', 'Weight5', 'Asian', 'Black.or.African.American',
                          'Unknown.Race', 'X.1..1', 'X.1..3', 'X.2..2', 'X.2..3', 'X.3..3', 'Unknown.Cyp2C9',
                          'VKORC1.A.G',
                          'VKORC1.A.A', 'VKORC1.Missing', 'Enzyme.Inducer', 'Amiodarone..Cordarone.'],
        'Athey_v1_500': ['V1.1', 'V1.2', 'V1.3', 'V1.4', 'V1.5', 'V1.6', 'V1.7', 'V1.8', 'V1.9', 'V1.10', 'V2.1',
                         'V2.2', 'V2.3', 'V2.4', 'V2.5', 'V2.6', 'V2.7', 'V2.8', 'V2.9', 'V2.10'],
        'Athey_v2_4000': ['V1', 'V2', 'V3']}

    data_group_true_outcome_cols_dict = {
        'Warfarin_3000': ['y0', 'y1', 'y2'],
        'Athey_v1_500': ['y0', 'y1'],
        'Athey_v2_4000': ['y0', 'y1']
    }

    data_group_ml_outcome_cols_dict = {}

    if ml == 'ml':
        data_group_ml_outcome_cols_dict = {
            'Warfarin_3000': ['ml0', 'ml1', 'ml2']
        }
    elif ml == 'linear':
        data_group_ml_outcome_cols_dict = {
            'Athey_v1_500': ['linear0', 'linear1'],
            'Athey_v2_4000': ['linear0', 'linear1']
        }
    elif ml == 'lasso':
        data_group_ml_outcome_cols_dict = {
            'Athey_v1_500': ['lasso0', 'lasso1'],
            'Athey_v2_4000': ['lasso0', 'lasso1']
        }

    data_path = os.getcwd() + data_path_dict[data_group][0]

    data_train = pd.read_csv(data_path + training_file)
    data_test = pd.read_csv(data_path + test_file)

    ##########################################################
    # output setting
    ##########################################################
    approach_name = ''
    if robust:
        approach_name = 'Robust'
    else:
        approach_name = 'Direct'
    out_put_name = training_file.split('.csv')[0] + '_' + approach_name + '_d_' + str(depth) + '_t_' + str(
        time_limit) + '_branching_limit_' + str(
        branching_limit) + '_pred_' + str(prob_type_pred)
    out_put_path = os.getcwd() + data_path_dict[data_group][1]
    sys.stdout = logger.logger(out_put_path + out_put_name + '.txt')

    ##########################################################
    # DataSet specific settings
    ##########################################################

    if 'Warfarin' in data_group:
        features = data_group_features_dict['Warfarin_3000']
        true_outcome_cols = data_group_true_outcome_cols_dict['Warfarin_3000']
        regression = data_group_ml_outcome_cols_dict['Warfarin_3000']
    else:
        features = data_group_features_dict[data_group]
        true_outcome_cols = data_group_true_outcome_cols_dict[data_group]
        regression = data_group_ml_outcome_cols_dict[data_group]
    treatment_col = 't'  # Name of the column in the dataset representing the treatment assigned to each data point
    outcome = 'y'


    prob_t = ''
    if prob_type_pred == 'tree':
        prob_t = 'prob_t_pred_tree'
    elif prob_type_pred == 'true':
        prob_t = 'prob_t'
    elif prob_type_pred == 'log':
        prob_t = 'prob_t_pred_log'


    ##########################################################
    # Creating and Solving the problem
    ##########################################################
    tree = Tree.Tree(depth)  # Tree structure: We create a complete binary tree of depth d
    primal = Primal.Primal(data_train, features, treatment_col, true_outcome_cols, outcome, regression, prob_t, robust,
                           tree, branching_limit,
                           time_limit)

    start_time = time.time()
    primal.create_primal_problem()
    primal.model.update()
    primal.model.optimize()
    end_time = time.time()
    solving_time = end_time - start_time

    ##########################################################
    # Preparing the output
    ##########################################################
    print("\n\n")
    print_tree(primal, tree,
               primal.model.getAttr("X", primal.b),
               primal.model.getAttr("X", primal.w),
               primal.model.getAttr("X", primal.p))
    print('\n\nTotal Solving Time', solving_time)
    print("obj value", primal.model.getAttr("ObjVal"))

    ##########################################################
    # Evaluation
    ##########################################################
    regret_train, best_found_train, treatment_classification_acc_train = get_metrics(primal, tree,
                                                                                     primal.model.getAttr("X",
                                                                                                          primal.b),
                                                                                     primal.model.getAttr("X",
                                                                                                          primal.w),
                                                                                     primal.model.getAttr("X",
                                                                                                          primal.p),
                                                                                     data_train, true_outcome_cols,
                                                                                     treatment_col)
    regret_test, best_found_test, treatment_classification_acc_test = get_metrics(primal, tree,
                                                                                  primal.model.getAttr("X", primal.b),
                                                                                  primal.model.getAttr("X", primal.w),
                                                                                  primal.model.getAttr("X", primal.p),
                                                                                  data_test, true_outcome_cols,
                                                                                  treatment_col)
    print("Policy Regret train (Sum)", regret_train)
    print("Best Treatment Found train (%)", best_found_train)
    print("treatment classification acc train (%)", treatment_classification_acc_train)
    print("Policy Regret test (Sum)", regret_test)
    print("Best Treatment Found test (%)", best_found_test)
    print("treatment classification acc test (%)", treatment_classification_acc_test)

    ##########################################################
    # writing info to the file
    ##########################################################
    #primal.model.write(out_put_path + out_put_name + '.lp')
    result_file = out_put_name + '.csv'
    with open(out_put_path + result_file, mode='a') as results:
        results_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        results_writer.writerow(
            [approach_name, training_file.split('.csv')[0], len(data_train),
             depth, branching_limit, time_limit,
             primal.model.getAttr("Status"),
             primal.model.getAttr("ObjVal"),
             primal.model.getAttr("MIPGap") * 100,
             solving_time,
             regret_train,
             best_found_train,
             treatment_classification_acc_train,
             regret_test,
             best_found_test,
             treatment_classification_acc_test,
             prob_type_pred, ml
             ])


if __name__ == "__main__":
    main(sys.argv[1:])
