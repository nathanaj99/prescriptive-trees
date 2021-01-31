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

datasets = [1, 2, 3, 4, 5]
probs = [0.10, 0.60, 0.85]
for prob in probs:
    for dataset in datasets:
        file_name = 'data_train_' + str(dataset) + '.csv'
        file_name_enc = 'data_train_enc_' + str(dataset) + '.csv'
        # ----- CHANGE THE FILE PATH -----
        file_path = '../data/Warfarin/Warfarin_' + str(prob) + '_2000/'
        df = pd.read_csv(file_path + file_name)
        df_enc = pd.read_csv(file_path + file_name_enc)
        t_unique = df['t'].unique()
        test = {}

        # for 0.1, y=1 will be minority class and for 0.85, y=0 will be minority class.
        # perform smote to increase the accuracy?

        for i in t_unique:
            buffer = df[df['t'] == i]
            X = buffer.iloc[:, :15]
            y = buffer.iloc[:, 15]

            #lr = LogisticRegression().fit(X, y)

            #lr = DecisionTreeRegressor().fit(X, y)
            #lr = Lasso(alpha=0.1).fit(X, y)
            #lr = LinearRegression().fit(X, y)
            #lr = DecisionTreeClassifier().fit(X, y)
            smote = SMOTE(sampling_strategy=1.0)
            X, y = smote.fit_resample(X, y)
            lr = RandomForestClassifier().fit(X, y)
            test[i] = lr

        for i in range(3):
            model = test[i]
            X = df.iloc[:, :15]
            prediction = model.predict(X)
            real = df['y' + str(i)]
            tn, fp, fn, tp = skm.confusion_matrix(real, prediction).ravel()
            # print(tn, fp, fn, tp)
            tpr = tp / float(tp + fn)
            tnr = tn / float(tn + fp)
            print(tpr, tnr)
            print(skm.accuracy_score(real, prediction))

            df_enc['ml' + str(i)] = prediction

        print(df_enc)

        df_enc.to_csv(file_path + file_name_enc, index=False)


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
