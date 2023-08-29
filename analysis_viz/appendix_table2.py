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
df_buffer = pd.read_csv(f'../results/synthetic/compiled/our_method.csv')
df_buffer['method'] = df_buffer['method'].map({'Direct': 'DM', 'Robust': 'DR', 'IPW': 'IPW'})
df_buffer = df_buffer[((df_buffer['budget'].isna()) | (df_buffer['budget'] == 1.0)) & (df_buffer['depth'] == 1)]
# print(df_buffer['ml'].value_counts())
# dm = df_buffer[(df_buffer['method'] == 'DM') & (df_buffer['ml'] == 'linear')]
# dr = df_buffer[(df_buffer['method'] == 'DR') & ((df_buffer['ml'] == 'linear') & (df_buffer['prop_pred'] == 'tree'))]
# ipw = df_buffer[(df_buffer['method'] == 'IPW') & (df_buffer['prop_pred'] == 'tree')]

df_buffer['prop_pred'] = df_buffer['prop_pred'].map({'tree': 'DT', 'log': 'Log'})
df_buffer['ml'] = df_buffer['ml'].map({'linear': 'LR', 'lasso': 'Lasso'})

def transform_model(row):
    if row['method'] == 'DM':
        return row['ml']
    elif row['method'] == 'IPW':
        return row['prop_pred']
    else:
        return f'{row["prop_pred"]}, {row["ml"]}'
    
df_buffer['model'] = df_buffer.apply(lambda row: transform_model(row), axis=1)

# df_buffer = pd.concat([ipw, dm, dr], ignore_index=True)

df = pd.concat([df, df_buffer[['depth', 'method', 'model', 'gap', 
                               'solve_time', 'regret_test', 'best_found_test']]], ignore_index=True)


# K-PT/B-PT
df_buffer = pd.read_csv(f'../results/synthetic/compiled/KB.csv')
df_buffer['method'] = df_buffer['method'].map({'Kallus': 'K-PT', 'Bertsimas': 'B-PT'})
df_buffer = df_buffer[df_buffer['depth'] == 1]
df_buffer['model'] = '-'
df = pd.concat([df, df_buffer[['depth', 'method', 'model', 'gap', 
                               'solve_time', 'regret_test', 'best_found_test']]], ignore_index=True)

# policytree
df_buffer = pd.read_csv(f'../results/synthetic/compiled/policytree/raw.csv')
for col, oosp, regret in zip([f'time_{i}' for i in ['0.1', '0.25', '0.5', '0.75', '0.9']],
                            [f'p{i}' for i in ['0.1', '0.25', '0.5', '0.75', '0.9']],
                            [f'oosr_{i}' for i in ['0.1', '0.25', '0.5', '0.75', '0.9']]):
    h = pd.DataFrame({'solve_time': df_buffer[col].tolist(),
                     'regret_test': df_buffer[regret].tolist(),
                    'best_found_test': df_buffer[oosp].tolist()})
    h['method'] = 'PT'
    h['best_found_test'] *= 100
    h['gap'] = 0
    h['depth'] = 1
    h['model'] = 'DT, LR'
    df = pd.concat([df, h], ignore_index=False)
    
    
# CF, CT
for m, m_name in zip(['cf', 'ct'], ['CF', 'CT']):
    df_buffer = pd.read_csv(f'../results/synthetic/compiled/CF/{m}_raw.csv')
#     df_trans = pd.DataFrame(columns=['method', 'randomization', 'realized_outcome_oos'])
    for col, oosp, regret in zip([f'time_{i}' for i in ['0.1', '0.25', '0.5', '0.75', '0.9']],
                            [f'p{i}' for i in ['0.1', '0.25', '0.5', '0.75', '0.9']],
                            [f'oosr_{i}' for i in ['0.1', '0.25', '0.5', '0.75', '0.9']]):
        h = pd.DataFrame({'solve_time': df_buffer[col].tolist(),
                     'regret_test': df_buffer[regret].tolist(),
                    'best_found_test': df_buffer[oosp].tolist()})
        h['method'] = m_name
        h['gap'] = 0
        h['best_found_test'] *= 100
        h['depth'] = '-'
        h['model'] = '-'
        df = pd.concat([df, h], ignore_index=False)
    
    
# RC
fp = '../results/synthetic/compiled/RC'
df_buffer = pd.read_csv(os.path.join(fp, 'raw.csv'))
df_buffer = df_buffer[df_buffer['method'] == 'lr']
df_buffer['method'] = 'R&C'
df_buffer = df_buffer.rename(columns={'time_elapsed': 'solve_time', 'oosp': 'best_found_test',
                                     'oos_regret': 'regret_test'})
df_buffer['gap'] = 0
df_buffer['depth'] = '-'
df_buffer['model'] = 'LR'
df_buffer['best_found_test'] *= 100
df = pd.concat([df, df_buffer[['depth', 'method', 'model', 'gap', 
                               'solve_time', 'regret_test', 'best_found_test']]], ignore_index=False)


mean_df = df.groupby(['depth', 'method', 'model']).agg('mean').reset_index().round(2)
std_df = df.groupby(['depth', 'method', 'model']).agg('std').reset_index().round(2)

combined = mean_df.merge(std_df, on=['depth', 'method', 'model'])
for col in ['gap', 'solve_time', 'regret_test', 'best_found_test']:
    combined[col] = combined.apply(lambda row: f'{row[f"{col}_x"]:.2f} ± {row[f"{col}_y"]:.2f}', axis=1)
    combined = combined.drop(columns=[f'{col}_{i}' for i in ['x', 'y']])

mapping = {'IPW': 1, 'DM': 2, 'DR': 3, 'K-PT': 4, 'B-PT': 5, 'PT': 6, 'CF': 0, 'R&C': 0, 'CT': 0}

combined['method_map'] = combined['method'].apply(lambda x: mapping[x])

print(combined.sort_values(by=['depth', 'method_map']).drop(columns=['method_map']).to_latex(index=False))