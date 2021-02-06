from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as skm
from imblearn.over_sampling import SMOTE
import random
import numpy as np

def random_minority(num, prediction, prob):
    for i in range(len(prediction)):
        if prediction[i] == num:
            prediction[i] = np.random.binomial(n=1, p=prob)
    return prediction


datasets = [1, 2, 3, 4, 5]

def warfarin():
    probs = [0.1, 0.33, 0.6, 0.85]
    for prob in probs:
        for dataset in datasets:
            file_name = 'data_train_' + str(dataset) + '.csv'
            file_name_enc = 'data_train_enc_' + str(dataset) + '.csv'
            # ----- CHANGE THE FILE PATH -----
            file_path = '../data/Warfarin2/Warfarin_' + str(prob) + '_2000/'
            df = pd.read_csv(file_path + file_name)
            df_enc = pd.read_csv(file_path + file_name_enc)
            t_unique = df['t'].unique()
            test = {}

            # for 0.1, y=1 will be minority class and for 0.85, y=0 will be minority class.
            # perform smote to increase the accuracy?

            for i in t_unique:
                buffer = df[df['t'] == i]
                X = buffer.iloc[:, :17]
                y = buffer.iloc[:, 17]
                print(y.value_counts())
                y_values = y.value_counts()

                #lr = LogisticRegression().fit(X, y)

                #lr = DecisionTreeRegressor().fit(X, y)
                #lr = Lasso(alpha=0.1).fit(X, y)
                #lr = LinearRegression().fit(X, y)
                #lr = DecisionTreeClassifier().fit(X, y)
                if y_values[1] > 5 and y_values[0] > 5:
                    smote = SMOTE(sampling_strategy=1.0, k_neighbors=5)
                elif y_values[0] <= 5:
                    smote = SMOTE(sampling_strategy=1.0, k_neighbors=y_values[0] - 1)
                elif y_values[1] <= 5:
                    smote = SMOTE(sampling_strategy=1.0, k_neighbors=y_values[1]-1)
                X, y = smote.fit_resample(X, y)
                lr = RandomForestClassifier().fit(X, y)
                test[i] = lr

            for i in range(3):
                model = test[i]
                X = df.iloc[:, :17]
                prediction = model.predict(X)
                real = df['y' + str(i)]
                tn, fp, fn, tp = skm.confusion_matrix(real, prediction).ravel()
                #print(tn, fp, fn, tp)
                tpr = tp / float(tp + fn)
                tnr = tn / float(tn + fp)
                print(tpr, tnr)
                """if i == 2 or i == 1:
                    prediction = df.apply(lambda x: 1 if x['t'] == i else 0, axis=1)
                    tn, fp, fn, tp = skm.confusion_matrix(real, prediction).ravel()
                    # print(tn, fp, fn, tp)
                    tpr = tp / float(tp + fn)
                    tnr = tn / float(tn + fp)
                    print("REVISED:")
                    print(tpr, tnr)"""


                print(skm.accuracy_score(real, prediction))

                #handling the minority class


                df_enc['ml' + str(i)] = prediction

            #print(df_enc)

            df_enc.to_csv(file_path + file_name_enc, index=False)


def v1():
    probs = [0.1, 0.25, 0.5, 0.75, 0.9]
    for prob in probs:
        for dataset in datasets:
            file_name = 'data_train_' + str(prob) + '_' + str(dataset) + '.csv'
            file_name_enc = 'data_train_enc_' + str(prob) + '_' + str(dataset) + '.csv'
            # ----- CHANGE THE FILE PATH -----
            file_path = '../data/Athey_v1_N_500/'
            df = pd.read_csv(file_path + file_name)
            df_enc = pd.read_csv(file_path + file_name_enc)
            t_unique = df['t'].unique()
            test_linear = {}
            test_lasso = {}

            # for 0.1, y=1 will be minority class and for 0.85, y=0 will be minority class.
            # perform smote to increase the accuracy?

            for i in t_unique:
                buffer = df[df['t'] == i]
                X = buffer.iloc[:, :2]
                y = buffer['y']

                # lr = LogisticRegression().fit(X, y)

                # lr = DecisionTreeRegressor().fit(X, y)
                lasso = Lasso(alpha=0.1).fit(X, y)
                linear = LinearRegression().fit(X, y)
                # lr = DecisionTreeClassifier().fit(X, y)
                test_linear[i] = linear
                test_lasso[i] = lasso

            for i in range(len(t_unique)):
                model_linear = test_linear[i]
                model_lasso = test_lasso[i]

                X = df.iloc[:, :2]
                prediction_linear = model_linear.predict(X)
                prediction_lasso = model_lasso.predict(X)

                real = df['y' + str(i)]

                print(skm.r2_score(real, prediction_linear))
                print(skm.r2_score(real, prediction_lasso))

                # handling the minority class

                df_enc['linear' + str(i)] = prediction_linear
                df_enc['lasso' + str(i)] = prediction_lasso

            print(df_enc)

            #df_enc.to_csv(file_path + file_name_enc, index=False)

def v2():
    probs = [0.1, 0.5, 0.75, 0.9]
    for prob in probs:
        for dataset in datasets:
            file_name = 'data_train_' + str(prob) + '_' + str(dataset) + '.csv'
            # ----- CHANGE THE FILE PATH -----
            file_path = '../data/Athey_v2_4000/'
            df = pd.read_csv(file_path + file_name)
            t_unique = df['t'].unique()
            test_linear = {}
            test_lasso = {}

            # for 0.1, y=1 will be minority class and for 0.85, y=0 will be minority class.
            # perform smote to increase the accuracy?

            for i in t_unique:
                buffer = df[df['t'] == i]
                X = buffer.iloc[:, :3]
                y = buffer['y']

                lasso = Lasso(alpha=0.1).fit(X, y)
                linear = LinearRegression().fit(X, y)
                # lr = DecisionTreeClassifier().fit(X, y)
                test_linear[i] = linear
                test_lasso[i] = lasso

            for i in range(len(t_unique)):
                model_linear = test_linear[i]
                model_lasso = test_lasso[i]

                X = df.iloc[:, :3]
                prediction_linear = model_linear.predict(X)
                prediction_lasso = model_lasso.predict(X)

                real = df['y' + str(i)]

                print(skm.r2_score(real, prediction_linear))
                print(skm.r2_score(real, prediction_lasso))

                # handling the minority class

                df['linear' + str(i)] = prediction_linear
                df['lasso' + str(i)] = prediction_lasso

            print(df)

            #df.to_csv(file_path + file_name_enc, index=False)

v2()
"""df = pd.read_csv('cleaned_IST1.csv')

df_train = df.iloc[:15200, :]
df_test = df.iloc[15200:, :]
print(df_train.shape)
print(df_test.shape)

df_test.to_csv('IST_reg_test.csv', index=False)

t_unique = df_train['t'].unique()
test = {}

for i in t_unique:
    buffer = df_train[df_train['t'] == i]
    X = buffer.iloc[:, :20]
    y = buffer.iloc[:, 20]
    lr = DecisionTreeRegressor().fit(X, y)
    #lr = Lasso(alpha=0.1).fit(X, y)
    #lr = LinearRegression().fit(X, y)
    test[i] = lr

for i in range(1, 7):
    model = test[i]
    X = df_train.iloc[:, :20]
    prediction = model.predict(X)
    df_train['reg' + str(i)] = prediction

df_train.to_csv('IST_reg_train.csv', index=False)"""
