from sklearn.linear_model import LinearRegression
import pandas as pd

datasets = [1, 2, 3, 4, 5]
probs = [0.5, 0.9]

for dataset in datasets:
    for prob in probs:
        file_name = 'data_train_' + str(prob) + '_' + str(dataset) + '.csv'
        file_name_enc = 'data_train_enc_' + str(prob) + '_' + str(dataset) + '.csv'
        df = pd.read_csv('../data/Athey_N_500/' + file_name)
        df_enc = pd.read_csv('../data/Athey_N_500/' + file_name_enc)
        t_unique = df['t'].unique()
        test = []

        for i in t_unique:
            buffer = df[df['t'] == i]
            X = buffer.iloc[:, :2]
            y = buffer.iloc[:, -2]
            lr = LinearRegression().fit(X, y)
            test.append(lr)

        index = 0
        for model in test:
            X = df.iloc[:, :2]
            prediction = model.predict(X)
            df_enc['reg'+str(index)] = prediction
            index += 1

        df_enc.to_csv('../data/direct_approach/' + file_name_enc)

