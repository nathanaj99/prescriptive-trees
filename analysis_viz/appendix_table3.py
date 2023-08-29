import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns
import matplotlib
import warnings
warnings.filterwarnings("ignore")

cols = ['method', 'file_name', 'num_rows',
                          'depth', 'branching_limit', 'time_limit',
                          'status', 'obj_value', 'gap', 'solve_time',
                          'regret_train', 'best_found_train', 'treatment_acc_train',
                          'regret_test', 'best_found_test', 'treatment_acc_test',
                          'prop_pred', 'ml', 'protected_col', 'fairness_bound', 'treatment_budget', 'budget']


df_buffer = pd.read_csv(f'../results/warfarin/compiled/dr_synthetic.csv', names=cols, header=None)
df_buffer1 = pd.read_csv(f'../results/synthetic/compiled/DR.csv')

df_buffer['randomization'] = df_buffer['file_name'].apply(lambda x: float(x.split('_')[-2]))
df_buffer['split'] = df_buffer['file_name'].apply(lambda x: int(x.split('_')[-1]))
df_buffer = df_buffer.rename(columns={'prop_pred': 'propensity_score_pred'})[['split', 'randomization', 'budget',
                                                                              'propensity_score_pred', 'ml',
                                                                             'solve_time']]

df_buffer = df_buffer1.merge(df_buffer)
df_buffer['gap'] = 0
df_buffer['method'] = 'DR'
df_buffer['ml'] = df_buffer['ml'].map({'linear': 'LR', 'lasso': 'Lasso'})
df_buffer['propensity_score_pred'] = df_buffer['propensity_score_pred'].map({'tree': 'DT', 'log': 'Log'})
df_buffer['model'] = df_buffer.apply(lambda row: f'{row["ml"]}, {row["propensity_score_pred"]}', axis=1)
df_buffer['oos_optimal_treatment'] *= 100

def budget_mapping(x):
    dic = {
        '0.05-0.09': [0.05, 0.09],
        '0.10-0.14': [0.10, 0.14],
        '0.15-0.19': [0.15, 0.19],
        '0.20-0.24': [0.20, 0.24],
        '0.25-0.29': [0.25, 0.29],
        '0.30-0.34': [0.30, 0.34],
        '0.35-0.40': [0.35, 0.40]
    }
    
    for k, v in dic.items():
        if x >= v[0] and x <= v[1]:
            return k

df_buffer['budget1'] = df_buffer['budget'].apply(lambda x: budget_mapping(x))

df_buffer = df_buffer[['tree_depth', 'method', 'model', 'budget1', 'gap', 'solve_time', 
                       'oos_regret', 'oos_optimal_treatment']]

mean_df = df_buffer.groupby(['tree_depth', 'method', 'model', 'budget1']).agg('mean').reset_index().round(2)
std_df = df_buffer.groupby(['tree_depth', 'method', 'model', 'budget1']).agg('std').reset_index().round(2)

combined = mean_df.merge(std_df, on=['tree_depth', 'method', 'model', 'budget1'])
for col in ['gap', 'solve_time', 'oos_regret', 'oos_optimal_treatment']:
    combined[col] = combined.apply(lambda row: f'{row[f"{col}_x"]:.2f} ± {row[f"{col}_y"]:.2f}', axis=1)
    combined = combined.drop(columns=[f'{col}_{i}' for i in ['x', 'y']])

print(combined.sort_values(by=['model', 'budget1']).to_latex(index=False))