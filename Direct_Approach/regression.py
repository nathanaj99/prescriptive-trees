from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from sklearn.model_selection import train_test_split

datasets = [1, 2, 3, 4, 5]

for dataset in datasets:
    file_name = 'data_train_' + str(dataset) + '.csv'
    #file_name_enc = 'data_train_' + str(dataset) + '.csv'
    df = pd.read_csv('../data/IST/' + file_name)
    #df_enc = pd.read_csv('../data/IST/' + file_name_enc)
    t_unique = df['t'].unique()
    test = {}

    for i in t_unique:
        buffer = df[df['t'] == i]
        X = buffer.iloc[:, :20]
        y = buffer.iloc[:, 20]
        lr = DecisionTreeRegressor().fit(X, y)
        #lr = Lasso(alpha=0.1).fit(X, y)
        #lr = LinearRegression().fit(X, y)
        test[i] = lr


    for i in range(1, 7):
        model = test[i]
        X = df.iloc[:, :20]
        prediction = model.predict(X)
        df['ml' + str(i)] = prediction

    df.to_csv('../data/IST/' + file_name, index=False)


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
