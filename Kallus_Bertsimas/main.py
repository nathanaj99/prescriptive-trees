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

def print_tree(branching, treatments, features):
    for i in branching:
        print('#########node', i)
        print(features[branching[i]])
    for i in treatments:
        print('#########node', i)
        print('leaf {}'.format(treatments[i]))


def datapoint_tree(node, i, test_X, test_real, test_t, branching, treatments, tree):
    if node in tree.Terminals:  # if datapoint has reached leaf node, calculate error
        index = treatments[node]
        ideal_outcome = max(test_real.iloc[i, :])
        difference = ideal_outcome - test_real.iloc[i, int(index)]
        # print(test_real.iloc[i, index])
        if difference == 0:
            count_optimal = 1
        else:
            count_optimal = 0

        if index == test_t[i]:
            same_treatment = 1
        else:
            same_treatment = 0
        return difference, count_optimal, same_treatment
    if test_X.iloc[i, branching[node]] <= 0:  # go left (node 2)
        return datapoint_tree(tree.get_left_children(node), i, test_X, test_real, test_t, branching, treatments, tree)
    else:
        return datapoint_tree(tree.get_right_children(node), i, test_X, test_real, test_t, branching, treatments, tree)

def get_metrics(test_X, test_real, test_t, branching, treatments, tree):
    difference = 0
    count_optimal = 0
    count_same = 0
    for i in range(len(test_X)):
        diff, optimal, treat = datapoint_tree(1, i, test_X, test_real, test_t, branching, treatments, tree)
        difference += diff
        count_optimal += optimal
        count_same += treat
    return difference, float(count_optimal)/len(test_X)*100, float(count_same)/len(test_X)*100


def main(argv):
    print(argv)

    training_file = None
    test_file = None
    depth = None
    branching_limit = None
    time_limit = None
    bertsimas = None
    n_min = None
    data_group = None

    try:
        opts, args = getopt.getopt(argv, "f:e:d:b:t:r:n:g:",
                                   ["training_file=", "test_file=", "depth=", "branching_limit=", "time_limit=",
                                    "bertsimas=", "n_min=", "data_group="])
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
        elif opt in ("-r", "--bertsimas"):
            bertsimas = arg
        elif opt in ("-n", "--n_min"):
            n_min = int(arg)
        elif opt in ("-g", "--data_group"):
            data_group = arg


    # ---- CHANGE FILE PATH ----
    if bertsimas == 'bertsimas':
        bertsimas = True
    elif bertsimas == 'kallus':
        bertsimas = False

    data_path_dict = {}
    if bertsimas:
        data_path_dict = {'Warfarin_3000': ('/../data/Warfarin/3000/', '/../Results_Bert/Warfarin/3000/'),
                          'Athey_v1_500': ('/../data/Athey_v1/500/', '/../Results_Bert/Athey_v1/500/'),
                          'Athey_v2_4000': ('/../data/Athey_v2/4000/', '/../Results_Bert/Athey_v2/4000/')}
    else:
        data_path_dict = {'Warfarin_3000': ('/../data/Warfarin/3000/', '/../Results_Kallus/Warfarin/3000/'),
                          'Athey_v1_500': ('/../data/Athey_v1/500/', '/../Results_Kallus/Athey_v1/500/'),
                          'Athey_v2_4000': ('/../data/Athey_v2/4000/', '/../Results_Kallus/Athey_v2/4000/')}

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

    data_path = os.getcwd() + data_path_dict[data_group][0]

    data_train = pd.read_csv(data_path + training_file)
    data_test = pd.read_csv(data_path + test_file)

    ##########################################################
    # output setting
    #########################################################

    approach_name = ''
    if bertsimas:
        approach_name = 'Bertsimas'
    else:
        approach_name = 'Kallus'
    out_put_name = training_file.split('.csv')[0] + '_' + approach_name + '_d_' + str(depth) + '_t_' + str(
        time_limit) + '_branching_limit_' + str(
        branching_limit)
    out_put_path = os.getcwd() + data_path_dict[data_group][1]
    sys.stdout = logger.logger(out_put_path + out_put_name + '.txt')

    ##########################################################
    # DataSet specific settings
    ##########################################################
    features = data_group_features_dict[data_group]
    treatment_col = 't'  # Name of the column in the dataset representing the treatment assigned to each data point
    true_outcome_cols = data_group_true_outcome_cols_dict[data_group]
    outcome = 'y'
    ##########################################################
    # Creating and Solving the problem
    ##########################################################
    tree = Tree.Tree(depth)  # Tree structure: We create a complete binary tree of depth d
    primal = Primal.Primal(data_train, features, treatment_col, true_outcome_cols, outcome, bertsimas, n_min, tree,
                           branching_limit,
                           time_limit)
    """primal.create_primal_problem()
    primal.model.update()
    primal.model.optimize()
    print(primal.model.getAttr("X", primal.b))
    print(primal.model.getAttr("X", primal.zeta))
    print(primal.model.getAttr('X', primal.p))
    print(primal.model.getAttr('X', primal.w))"""


    start_time = time.time()
    primal.create_primal_problem()
    primal.model.update()
    primal.model.optimize()
    end_time = time.time()
    solving_time = end_time - start_time

    ##########################################################
    # Preparing the output
    ##########################################################
    # get branching and treatments
    g = primal.model.getAttr("X", primal.gamma).items()
    l = primal.model.getAttr("X", primal.lamb).items()

    branching = {i[0][0]: i[0][1] for i in g if i[1] == 1.0}
    treatments = {i[0][0]: i[0][1] for i in l if i[1] == 1.0}

    print("\n\n")
    print_tree(branching, treatments, features)
    print('\n\nTotal Solving Time', solving_time)
    print("obj value", primal.model.getAttr("ObjVal"))

    ##########################################################
    # Evaluation
    ##########################################################

    train_X = data_train[features]
    train_y = data_train[outcome]
    train_real = data_train[true_outcome_cols]
    train_t = data_train[treatment_col]
    test_X = data_test[features]
    test_y = data_test[outcome]
    test_real = data_test[true_outcome_cols]
    test_t = data_test[treatment_col]

    regret_train, best_found_train, treatment_classification_acc_train = get_metrics(train_X, train_real, train_t, branching, treatments, tree)
    regret_test, best_found_test, treatment_classification_acc_test = get_metrics(test_X, test_real, test_t, branching, treatments, tree)

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
             ])


if __name__ == "__main__":
    main(sys.argv[1:])
