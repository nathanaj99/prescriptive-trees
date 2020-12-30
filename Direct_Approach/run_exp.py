import main as primal
import os

depths = [0,1,2]
samples = [1, 2, 3, 4, 5]
probs = [0.1, 0.5, 0.9]

"""for s in samples:
    for d in depths:
        for prob in probs:
            training_file = 'data_train_enc_' + str(prob) + '_' + str(s) + '.csv'
            test_file = 'data_test_enc_' + str(prob) + '_' + str(s) + '.csv'
            primal.main(["-f", training_file, "-e", test_file, "-d", d, "-b", 100, "-t", 3600, "-p", 1, "-r", True])"""

primal.main(["-f", 'cleaned_IST_reg.csv', "-e", 'data_test_0.5_1.csv', "-d", 2, "-b", 100, "-t", 600, "-p", 0, "-r", False])