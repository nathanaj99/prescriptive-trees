import main as primal
import os

samples = [1, 2, 3, 4, 5]
robust = ['robust', 'direct']
models = ["linear", "lasso"]

# # Warfarin_3000 ::
# DIRECT & ROBUST: 5(sample)*4(depth)*4(randomization) = 80 instances
for s in samples:
    for d in [1, 2, 3, 4]:
        for threshold in [0.1, 0.33, 0.6, 0.85]:
            for rob in robust:
                training_file = 'data_train_enc_' + str(threshold) + '_' + str(s) + '.csv'
                test_file = 'data_test_enc_' + str(threshold) + '_' + str(s) + '.csv'
                primal.main(["-f", training_file, "-e", test_file, "-d", d, "-b", 100, "-t", 3600, "-p", 'tree', "-r",
                             rob, "-g", "Warfarin_3000", "-m", "ml"])


# Athey_v1_500 ::
# DIRECT: 5(sample)*4(depth)*5(randomization)*2(models) = 200 instances
# ROBUST: 5(sample)*4(depth)*5(randomization)*2(models)*3(prob_score) = 600 instances
for s in samples:
    for d in [1, 2, 3, 4]:
        for threshold in [0.1, 0.25, 0.5, 0.75, 0.9]:
            for ml in models:
                training_file = 'data_train_enc_' + str(threshold) + '_' + str(s) + '.csv'
                test_file = 'data_test_enc_' + str(threshold) + '_' + str(s) + '.csv'
                for rob in robust:
                    if rob: # if robust, then execute all three
                        for pred in ['true', 'tree', 'log']:
                            primal.main(["-f", training_file, "-e", test_file, "-d", d, "-b", 100, "-t", 3600, "-p", pred,
                                         "-r", rob, "-g", "Athey_v1_500", "-m", ml])
                    else:
                        primal.main(["-f", training_file, "-e", test_file, "-d", d, "-b", 100, "-t", 3600, "-p", "tree",
                                     "-r", rob, "-g", "Athey_v1_500", "-m", ml])


# Athey_v2_4000 ::
# # ROBUST: 5(sample)*2(depth)*5(randomization)*3(prob_score)*2(models) = 300 instances
# DIRECT: 5(sample)*2(depth)*5(randomization)*2(models) = 100 instances
for s in samples:
    for d in [1, 2]:
        for threshold in [0.1, 0.25, 0.5, 0.75, 0.9]:
            for ml in models:
                training_file = 'data_train_' + str(threshold) + '_' + str(s) + '.csv'
                test_file = 'data_test_' + str(threshold) + '_' + str(s) + '.csv'
                for rob in robust:
                    if rob: # if robust, then execute all three
                        for pred in ['true', 'tree', 'log']:
                            primal.main(["-f", training_file, "-e", test_file, "-d", d, "-b", 100, "-t", 3600, "-p", pred,
                                         "-r", rob, "-g", "Athey_v2_4000", "-m", ml])
                    else:
                        primal.main(["-f", training_file, "-e", test_file, "-d", d, "-b", 100, "-t", 3600, "-p", "tree",
                                     "-r", rob, "-g", "Athey_v2_4000", "-m", ml])