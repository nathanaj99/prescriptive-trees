import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns
import matplotlib
import warnings
warnings.filterwarnings("ignore")

df = pd.DataFrame()
             
# our method
df_buffer = pd.read_csv(f'../results/synthetic/compiled/our_method.csv')

df_buffer = df_buffer[((df_buffer['method'] == 'IPW') & (df_buffer['prop_pred'] == 'tree')) |\
                     ((df_buffer['method'] == 'Direct') & (df_buffer['ml'] == 'linear')) |\
                     ((df_buffer['method'] == 'Robust') & (df_buffer['prop_pred'] == 'tree') & (df_buffer['ml'] == 'linear'))]

df_buffer = df_buffer[((df_buffer['budget'].isna()) | (df_buffer['budget'] == 1)) & (df_buffer['depth'] == 1)]

df_buffer['method'] = df_buffer['method'].map({'Direct': 'DM', 'Robust': 'DR', 'IPW': 'IPW'})
df = pd.concat([df, df_buffer[['method', 'gap', 'solve_time']]], ignore_index=True)


# kallus bertsimas
df_buffer = pd.read_csv(f'../results/synthetic/compiled/KB.csv')
df_buffer['method'] = df_buffer['method'].map({'Kallus': 'K-PT', 'Bertsimas': 'B-PT'})
df = pd.concat([df, df_buffer[['method', 'gap', 'solve_time']]], ignore_index=True)

# policytree
df_buffer = pd.read_csv(f'../results/synthetic/compiled/policytree/raw.csv')
for col, name in zip(['time_0.1', 'time_0.25', 'time_0.5', 'time_0.75', 'time_0.9'], ['0.1', '0.25', '0.5', '0.75', '0.9']):
    h = pd.DataFrame({'solve_time': df_buffer[col].tolist()})
    h['method'] = 'PT'
    h['gap'] = 0
    df = pd.concat([df, h], ignore_index=False)

# CF, CT
for m, m_name in zip(['cf', 'ct'], ['CF', 'CT']):
    df_buffer = pd.read_csv(f'../results/synthetic/compiled/CF/{m}_raw.csv')
    for col, name in zip(['time_0.1', 'time_0.25', 'time_0.5', 'time_0.75', 'time_0.9'], ['0.1', '0.25', '0.5', '0.75', '0.9']):
        h = pd.DataFrame({'solve_time': df_buffer[col].tolist()})
        h['method'] = m_name
        h['gap'] = 0
        df = pd.concat([df, h], ignore_index=False)
        
# rc
fp = '../results/synthetic/compiled/RC'
df_buffer = pd.read_csv(os.path.join(fp, 'raw.csv'))
df_buffer = df_buffer[df_buffer['method'] == 'lr']
df_buffer['method'] = 'RC'
df_buffer = df_buffer.rename(columns={'time_elapsed': 'solve_time'})
df_buffer['gap'] = 0
df = pd.concat([df, df_buffer[['method', 'gap', 'solve_time']]], ignore_index=False)


from matplotlib.patches import Patch
from matplotlib.lines import Line2D
colors ={"IPW (DT)": "#FFC7C7", "DM (LR)": "#FFA4A4", "DR (LR, DT)": "#EB6262",
         "K-PT": "#BFBFBF", "B-PT": "#878787", 'PT': "#CCDA2F", 'CF': "#9CCF9B", 'CT': "#21C51D", 'R&C (LR)': '#B1B0CD'}
legend_elements = [Line2D([0], [0], color=v, lw=2, linestyle='-', label=k) for k, v in colors.items()]

matplotlib.rcParams.update({'font.size': 24})
colors ={"IPW": "#FFC7C7", "DM": "#FFA4A4", "DR": "#EB6262", 
         "K-PT": "#BFBFBF", "B-PT": "#878787", 'PT': "#CCDA2F", 'CF': "#9CCF9B", 'CT': "#21C51D", 'RC': '#B1B0CD'}
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 7), sharey=True)
for method in ['IPW', 'DM', 'DR', 'K-PT', 'B-PT', 'PT', 'CF', 'CT', 'RC']:
    values, base = np.histogram(df[df['method'] == method]['solve_time'], bins=40)
    cumulative = np.cumsum(values)
    int_max = cumulative[-1]
    cumulative = np.insert(cumulative, 0, 0, axis=0)
    cumulative = np.append(cumulative, int_max)
    base = np.insert(base, 0, 0, axis=0)

    ax.plot(base, cumulative, c=colors[method], label=method, linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('# Instances Solved')
    ax.legend(handles=legend_elements, bbox_to_anchor=(1, 1.0), prop={'size': 20}, ncol=1)

plt.grid(True)
plt.savefig('figs/synthetic_comp_times.pdf', bbox_inches='tight')
plt.show()