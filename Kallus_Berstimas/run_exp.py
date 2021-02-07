import main as primal
import os

depths = [0, 1, 2]
samples = [1]

for s in samples:
    for d in depths:
        training_file = 'data_train_enc_' + str(s) + '.csv'
        test_file = 'data_test_enc_' + str(s) + '.csv'
        # Kallus
        primal.main(["-f", training_file, "-e", test_file, "-d", d, "-b", 100, "-t", 3600, "-r", False, "-n", 0]) # p always 1 for IST

        # DOUBLY ROBUST
        #primal.main(["-f", training_file, "-e", test_file, "-d", d, "-b", 100, "-t", 3600, "-r", True, "-n", 0])

#primal.main(["-f", 'data_train_enc_1.csv', "-e", 'data_test_enc_1.csv', "-d", 2, "-b", 100, "-t", 3600, "-p", 1, "-r", False])