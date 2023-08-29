import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns
import matplotlib
import warnings
warnings.filterwarnings("ignore")

def transform_model(row):
    if row['method'] == 'DM':
        return row['ml']
    elif row['method'] == 'IPW':
        return row['prop_pred']
    else:
        return f'{row["prop_pred"]}, {row["ml"]}'

df = pd.DataFrame()
             
# our method
df_buffer = pd.read_csv(f'../results/warfarin/compiled/unconstrained_agg.csv')
df_buffer['method'] = df_buffer['method'].map({'Direct': 'DM', 'Robust': 'DR', 'IPW': 'IPW'})
df_buffer['prop_pred'] = df_buffer['prop_pred'].map({'tree': 'DT'})
df_buffer['ml'] = 'RF/Log'

df_buffer['model'] = df_buffer.apply(lambda row: transform_model(row), axis=1)
df = pd.concat([df, df_buffer[['depth', 'method', 'model', 'gap', 
                               'solve_time', 'regret_test', 'best_found_test']]], ignore_index=True)

# kallus bertsimas
df_buffer = pd.read_csv(f'../results/warfarin/compiled/KB.csv')
df_buffer['method'] = df_buffer['method'].map({'Kallus': 'K-PT', 'Bertsimas': 'B-PT'})
df_buffer['model'] = '-'
df = pd.concat([df, df_buffer[['depth', 'method', 'model', 'gap', 
                               'solve_time', 'regret_test', 'best_found_test']]], ignore_index=True)

# PT
df_buffer = pd.read_csv(f'../results/warfarin/compiled/policytree/raw_proba.csv')
for col, oosp, regret in zip(['random_time', 'r0.06_time', 'r0.11_time'], ['random', 'r0.06', 'r0.11'],
                            ['random_oos_regret', 'r0.06_oos_regret', 'r0.11_oos_regret']):
    h = pd.DataFrame({'solve_time': df_buffer[col].tolist(),
                    'regret_test': df_buffer[regret].tolist(),
                    'best_found_test': df_buffer[oosp].tolist()})
    h['method'] = 'PT'
    h['gap'] = 0
    h['best_found_test'] *= 100
    h['depth'] = 2
    h['model'] = 'DT, Mixed'
    df = pd.concat([df, h], ignore_index=False)
    
# CF, CT
for m, m_name in zip(['cf', 'cf_untuned', 'ct'], ['CF', 'CF (untuned)', 'CT']):
    df_buffer = pd.read_csv(f'../results/warfarin/compiled/CF/{m}_baseline_raw.csv')
    for col, oosp, regret, in zip(['time_random', 'time_r0.06', 'time_r0.11'], ['random', 'r0.06', 'r0.11'],
                            ['random_oos_regret', 'r0.06_oos_regret', 'r0.11_oos_regret']):
        h = pd.DataFrame({'solve_time': df_buffer[col].tolist(),
                         'regret_test': df_buffer[regret].tolist(),
                    'best_found_test': df_buffer[oosp].tolist()})
        h['method'] = m_name
        h['depth'] = '-'
        h['best_found_test'] *= 100
        h['gap'] = 0
        h['model'] = '-'
        df = pd.concat([df, h], ignore_index=False)
        
#RC
df_buffer = pd.read_csv(f'../results/warfarin/compiled/RC/rc_raw.csv')
df_buffer_random = df_buffer[df_buffer['randomization'] == '0.33']
df_buffer_random1 = df_buffer_random[df_buffer_random['model'] == 'balanced_rf']
df_buffer_random1['model'] = 'best'
df_buffer_random = pd.concat([df_buffer_random[df_buffer_random['model'] != 'lrrf'], df_buffer_random1], ignore_index=True)
df_buffer_random['model'] = df_buffer_random['model'].map({'balanced_rf': 'RF', 'best': 'Best',
                                                          'balanced_lr': 'Log'})

df_buffer_other = df_buffer[df_buffer['randomization'] != '0.33']
df_buffer_other['model'] = df_buffer_other['model'].map({'balanced_rf': 'RF', 'lrrf': 'Best',
                                                          'balanced_lr': 'Log'})

df_buffer = pd.concat([df_buffer_random, df_buffer_other], ignore_index=True).rename(columns={'oos_regret': 'regret_test',
                                                                                              'oosp': 'best_found_test'})
df_buffer['method'] = 'R&C'
df_buffer['gap'] = 0
df_buffer['depth'] = '-'
df_buffer['best_found_test'] *= 100

df_buffer = df_buffer.drop(columns=['randomization', 'dataset', 'seed'])
df = pd.concat([df, df_buffer], ignore_index=False)


mean_df = df.groupby(['depth', 'method', 'model']).agg('mean').reset_index().round(2)
std_df = df.groupby(['depth', 'method', 'model']).agg('std').reset_index().round(2)

combined = mean_df.merge(std_df, on=['depth', 'method', 'model'])

for col in ['gap', 'solve_time', 'regret_test', 'best_found_test']:
    combined[col] = combined.apply(lambda row: f'{row[f"{col}_x"]:.2f} ± {row[f"{col}_y"]:.2f}', axis=1)
    combined = combined.drop(columns=[f'{col}_{i}' for i in ['x', 'y']])


mapping = {'IPW': 1, 'DM': 2, 'DR': 3, 'K-PT': 4, 'B-PT': 5, 'PT': 6, 'CF': 0, 'CF (untuned)': 0, 'CT': 0, 'R&C': 0}

combined['method_map'] = combined['method'].apply(lambda x: mapping[x])
print(combined.sort_values(by=['depth', 'method_map']).drop(columns=['method_map']).to_latex(index=False))








