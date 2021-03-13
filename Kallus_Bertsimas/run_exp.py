import main as primal
import os

samples = [1, 2, 3, 4, 5]
bertsimas = ['bertsimas', 'kallus']

# # Warfarin_3000 :: 5(sample)*4(depth)*4(randomization) = 80 instances
# for s in samples:
#     for d in [1, 2, 3, 4]:
#         for threshold in [0.1, 0.33, 0.6, 0.85]:
#             for bert in bertsimas:
#                 training_file = 'data_train_enc_' + str(threshold) + '_' + str(s) + '.csv'
#                 test_file = 'data_test_enc_' + str(threshold) + '_' + str(s) + '.csv'
#                 primal.main(["-f", training_file, "-e", test_file, "-d", d, "-b", 100, "-t", 3600, "-r", bert,
#                              "-n", 0, "-g", "Warfarin_3000"])


# training_file = 'data_train_enc_' + str(0.1) + '_' + str(1) + '.csv'
# test_file = 'data_test_enc_' + str(0.1) + '_' + str(1) + '.csv'
# primal.main(["-f", training_file, "-e", test_file, "-d", 1, "-b", 100, "-t", 3600, "-r", 'bertsimas',
#              "-n", 0, "-g", "Warfarin_3000"])


training_file = 'data_train_enc_' + str(0.6) + '_' + str(1) + '.csv'
test_file = 'data_test_enc_' + str(0.6) + '_' + str(1) + '.csv'
primal.main(["-f", training_file, "-e", test_file, "-d", 2, "-b", 100, "-t", 3600, "-r", 'kallus',
             "-n", 0, "-g", "Warfarin_3000"])


# Athey_v1_500 :: 5(sample)*4(depth)*5(randomization)*3(prob_score) = 300 instances
# for s in samples:
#     for d in [1, 2, 3, 4]:
#         for threshold in [0.1, 0.25, 0.5, 0.75, 0.9]:
#             for bert in bertsimas:
#                 training_file = 'data_train_enc_' + str(threshold) + '_' + str(s) + '.csv'
#                 test_file = 'data_test_enc_' + str(threshold) + '_' + str(s) + '.csv'
#                 primal.main(["-f", training_file, "-e", test_file, "-d", d, "-b", 100, "-t", 3600, "-r", bert,
#                              "-n", 0, "-g", "Athey_v1_500"])
#
# # Athey_v2_4000 :: 5(sample)*2(depth)*5(randomization) = 50 instances
# for s in samples:
#     for d in [1, 2]:
#         for threshold in [0.1, 0.25, 0.5, 0.75, 0.9]:
#             for bert in bertsimas:
#                 training_file = 'data_train_' + str(threshold) + '_' + str(s) + '.csv'
#                 test_file = 'data_test_' + str(threshold) + '_' + str(s) + '.csv'
#                 primal.main(["-f", training_file, "-e", test_file, "-d", d, "-b", 100, "-t", 3600, "-r", bert,
#                              "-n", 0, "-g", "Athey_v2_4000"])