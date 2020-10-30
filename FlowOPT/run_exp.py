import main as primal

depths = [0, 1, 2]
samples = [1, 2, 3, 4, 5]

# for s in samples:
#     for d in depths:
#         for pred in [0,1]:
#             training_file = 'data_train_enc_0.9_' + str(s) + '.csv'
#             test_file = 'data_test_enc_0.9_' + str(s) + '.csv'
#             primal.main(["-f", training_file, "-e", test_file, "-d", d, "-b", 100, "-t", 600, "-p", pred])
#


primal.main(["-f", 'data_train_enc_0.9_5.csv', "-e", 'data_test_enc_0.9_5.csv', "-d", 2, "-b", 100, "-t", 600, "-p", 1])