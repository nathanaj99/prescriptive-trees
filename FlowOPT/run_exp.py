import main as primal
import os

depths = [0, 1, 2]
samples = [1, 2, 3, 4, 5]


# for s in samples:
#     for d in depths:
#         for threshold in [0.1, 0.5, 0.6, 0.75, 0.9]:
#             for pred in [0, 1]:
#                 threshold_str = str(threshold)
#                 training_file = 'data_train_enc_' + str(threshold) + '_' + str(s) + '.csv'
#                 test_file = 'data_test_enc_' + str(threshold) + '_' + str(s) + '.csv'
#                 primal.main(["-f", training_file, "-e", test_file, "-d", d, "-b", 100, "-t", 3600, "-p", pred])

# primal.main(["-f", 'data_train_0.5_1.csv', "-e", 'data_test_0.5_1.csv', "-d", 1, "-b", 100, "-t", 600, "-p", 0])


for s in samples:
    for d in depths:
        training_file = 'data_train_'  + str(s) + '.csv'
        test_file = 'data_test_' +  str(s) + '.csv'
        primal.main(["-f", training_file, "-e", test_file, "-d", d, "-b", 100, "-t", 3600, "-p", 1])