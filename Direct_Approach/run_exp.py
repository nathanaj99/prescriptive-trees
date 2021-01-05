import main as primal
import os

depths = [0, 1, 2]
samples = [1, 2]

for s in samples:
    for d in depths:
        training_file = 'data_train_enc_' + str(s) + '.csv'
        test_file = 'data_test_enc_' + str(s) + '.csv'
        # DIRECT APPROACH
        primal.main(["-f", training_file, "-e", test_file, "-d", d, "-b", 100, "-t", 3600, "-p", 1, "-r", False]) # p always 1 for IST

        # DOUBLY ROBUST
        primal.main(["-f", training_file, "-e", test_file, "-d", d, "-b", 100, "-t", 3600, "-p", 1, "-r", True])