import pandas as pd

df = pd.read_csv('../data/Warfarin/Warfarin_0.85_2000/data_train_enc_1.csv')
print(df['t'].value_counts())

df_test = pd.read_csv('../data/Warfarin/Warfarin_0.85_2000/data_test_enc_1.csv')
print(df_test['t'].value_counts())

df2 = pd.read_csv('../data/Warfarin2/Warfarin_0.85_2000/data_train_enc_1.csv')
print(df2['t'].value_counts())

df_test2 = pd.read_csv('../data/Warfarin2/Warfarin_0.85_2000/data_test_enc_1.csv')
print(df_test2['t'].value_counts())
