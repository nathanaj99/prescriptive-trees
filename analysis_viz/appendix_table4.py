import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns
import matplotlib
import warnings
warnings.filterwarnings("ignore")

df_buffer = pd.read_csv(f'../results/warfarin/compiled/DM.csv')
cols = ['method', 'file_name', 'num_rows',
                          'depth', 'branching_limit', 'time_limit',
                          'status', 'obj_value', 'gap', 'solve_time',
                          'regret_train', 'best_found_train', 'treatment_acc_train',
                          'regret_test', 'best_found_test', 'treatment_acc_test',
                          'prop_pred', 'ml', 'protected_col', 'fairness_bound', 'treatment_budget', 'budget']


df_buffer1 = pd.read_csv('../results/warfarin/compiled/dm_warfarin.csv', header=None, names=cols)
df_buffer1 = df_buffer1[df_buffer1['protected_col'].notna()]
df_buffer1['randomization'] = df_buffer1['file_name'].apply(lambda x: x.split('_')[-2])
df_buffer1['split'] = df_buffer1['file_name'].apply(lambda x: int(x.split('_')[-1]))
df_buffer1['seed'] = [1]*120 + [2]*120 + [3]*120 + [4]*120 + [5]*120

df_buffer = df_buffer[df_buffer['fairness'] < 0.09]
df_buffer1 = df_buffer1.rename(columns={'fairness_bound': 'fairness'})
merged = df_buffer.merge(df_buffer1[['randomization', 'split', 'seed', 'fairness', 'solve_time']])
merged['oos_regret'] = 1386 - merged['realized_outcome_oos'] * 1386

merged['gap'] = 0
merged['model'] = 'RF/Log'

merged = merged[['tree_depth', 'method', 'model', 'fairness', 'gap', 'solve_time', 'dr_disparity', 'realized_disparity', 
                     'oos_regret', 'realized_outcome_oos']]


mean_df = merged.groupby(['tree_depth', 'method', 'model', 'fairness']).agg('mean').reset_index().round(2)
std_df = merged.groupby(['tree_depth', 'method', 'model', 'fairness']).agg('std').reset_index().round(2)

combined = mean_df.merge(std_df, on=['tree_depth', 'method', 'model', 'fairness'])
for col in ['gap', 'solve_time', 'dr_disparity', 'realized_disparity', 
                     'oos_regret', 'realized_outcome_oos']:
    combined[col] = combined.apply(lambda row: f'{row[f"{col}_x"]:.2f} ± {row[f"{col}_y"]:.2f}', axis=1)
    combined = combined.drop(columns=[f'{col}_{i}' for i in ['x', 'y']])

print(combined.sort_values(by=['fairness']).to_latex(index=False))