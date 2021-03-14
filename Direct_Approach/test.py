import pandas as pd
import numpy as np

for seed in [1, 2, 3, 4, 5]:
    for prob in [0.33, 'r0.11', 'r0.06']:
        df = pd.read_csv('seed' + str(seed) + '/warfarin_enc_' + str(prob) + '.csv')
        for t in [0, 1, 2]:
            print(seed, prob, t)
            df_buffer = df[df['t'] == int(t)]
            print(df_buffer['y'].value_counts())