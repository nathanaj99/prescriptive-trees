import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import scipy.stats
import time
import os


def run(df, df_test, model):
    t_unique = df['t'].unique()
    
    start = time.time()
    
    for t in t_unique:
        buffer = df[df['t'] == t]
        X = buffer.iloc[:, :2]
        y = buffer['y']

        lr = model.fit(X, y)

        X_test = df_test.iloc[:, :2]
        df_test['pred' + str(t)] = lr.predict(X_test)

    ## EVALUATE PERFORMANCE
    def find_highest_y(row):
        if row['pred1'] > row['pred0']:
            return 1
        else:
            return 0

    def t_opt(row):
        if row['y1'] > row['y0']:
            return 1
        else:
            return 0
        
    def expected_outcomes(row):
        if row['pred1'] > row['pred0']:
            return row['y1']
        else:
            return row['y0']

    df_test['t_opt'] = df_test.apply(lambda row: t_opt(row), axis=1) 
    df_test['y_opt'] = df_test.apply(lambda row: row['y1'] if row['y1'] > row['y0'] else row['y0'], axis=1) 
    
    df_test['t_pred'] = df_test.apply(lambda row: find_highest_y(row), axis=1)
    
    end = time.time()
    
    df_test['realized_y'] = df_test.apply(lambda row: expected_outcomes(row), axis=1)
    df_test['oos_regret'] = df_test['y_opt'] - df_test['realized_y']
    hi = df_test['t_opt'] == df_test['t_pred']

    return (df_test['t_opt'] == df_test['t_pred']).sum()/len(df_test), end-start, df_test['oos_regret'].sum()

def driver(model):
    probs = [0.1, 0.25, 0.5, 0.75, 0.9]
    datasets = [1, 2, 3, 4, 5]
    
    policy_opt_dic = {}
    times_dic = {}
    regret_dic = {}
    for prob in probs:
        policy_opt = []
        times = []
        oos_regret = []
        for dataset in datasets:
            fp = '../../data/processed/synthetic/'
            fn = f'data_train_{prob}_{dataset}.csv'
            fn_test = f'data_test_{prob}_{dataset}.csv'
            fn_enc = f'data_train_enc_{prob}_{dataset}.csv'

            df = pd.read_csv(os.path.join(fp, fn))
            df_test = pd.read_csv(os.path.join(fp, fn_test))
            if model is None:
                policy_opt.append(df_test['y0'].mean())
                df_test['y_opt'] = df_test.apply(lambda row: row['y1'] if row['y1'] > row['y0'] else row['y0'], axis=1) 
                regret = (df_test['y_opt'] - df_test['y0']).sum()
                oos_regret.append(regret)
            else:
                opt, time, regret = run(df, df_test, model)
                policy_opt.append(opt)
                times.append(time)
                oos_regret.append(regret)
        policy_opt_dic[prob] = policy_opt
        times_dic[prob] = times if model is not None else [np.nan]*5
        regret_dic[prob] = oos_regret
    return policy_opt_dic, times_dic, regret_dic

df = pd.DataFrame(columns=['method', 'prob_opt', 'oosp'])

def make_df(dics, method):
    prob_opt = []
    oosp_opt = []
    dataset = []
    times = []
    regret = []
#     print(dics)
    dic = dics[0]
    for k, v in dic.items():
        prob_opt += [k]*5
        dataset += [1, 2, 3, 4, 5]
        oosp_opt += v
        
        
    for k, v in dics[1].items():
        times += v
        
    for k, v in dics[2].items():
        regret += v
        
    df = pd.DataFrame({'method': [method]*25, 'dataset': dataset, 'prob_opt': prob_opt, 'oosp': oosp_opt, 'oos_regret': regret, 'time_elapsed': times})
    return df

# random assignment
df = pd.concat([df, make_df(driver(None), 'random')], ignore_index=True)

# lasso
df = pd.concat([df, make_df(driver(Lasso(alpha=0.08)), 'lasso')], ignore_index=True)

# linear regression
df = pd.concat([df, make_df(driver(LinearRegression()), 'lr')], ignore_index=True)


fp = '../../results/synthetic/compiled/'
df.to_csv(os.path.join(fp, 'raw.csv'), index=False)
