from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
import pandas as pd

datasets = [1, 2, 3, 4, 5]
probs = [0.1, 0.5, 0.9]

for dataset in datasets:
    for prob in probs:
        file_name = 'data_train_' + str(prob) + '_' + str(dataset) + '.csv'
        file_name_enc = 'data_train_enc_' + str(prob) + '_' + str(dataset) + '.csv'
        df = pd.read_csv('../data/Athey_v1_N_500/' + file_name)
        df_enc = pd.read_csv('../data/Athey_v1_N_500/' + file_name_enc)
        t_unique = df['t'].unique()
        test = {}

        for i in t_unique:
            buffer = df[df['t'] == i]
            X = buffer.iloc[:, :2]
            y = buffer.iloc[:, -3]
            #dr = DecisionTreeRegressor().fit(X, y)
            lr = Lasso(alpha=0.1).fit(X, y)
            #lr = LinearRegression().fit(X, y)
            test[i] = lr


        for i in range(2):
            model = test[i]
            X = df.iloc[:, :2]
            prediction = model.predict(X)
            df_enc['reg' + str(i)] = prediction

        df_enc.to_csv('../data/direct_v1_500/' + file_name_enc, index=False)