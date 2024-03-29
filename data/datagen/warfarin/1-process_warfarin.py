import pandas as pd
import math
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import collections
import sklearn.metrics as skm
import os


df = pd.read_csv('../raw/warfarin.csv')

def main(df, seed, path):
    # VCORC1 imputation rules
    impute = df['VKORC1 rs9923231'].isna()

    for index, row in df.iterrows():
        #print(type(row['VKORC1 QC genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T']))
        # determine if the rs9923231 row is missing
        if impute[index]:
            # check the QC column?
            if row['Race (OMB)'] in ['White', 'Asian']:
                if row['VKORC1 rs2359612'] == 'C/C':
                    df.loc[df.index[index], 'VKORC1 rs9923231'] = 'G/G'
                if row['VKORC1 rs2359612'] == 'T/T':
                    df.loc[df.index[index], 'VKORC1 rs9923231'] = 'A/A'
                if row['VKORC1 rs2359612'] == 'C/T':
                    df.loc[df.index[index], 'VKORC1 rs9923231'] = 'A/G'
                if row['VKORC1 rs8050894'] == 'G/G':
                    df.loc[df.index[index], 'VKORC1 rs9923231'] = 'G/G'
                if row['VKORC1 rs8050894'] == 'C/C':
                    df.loc[df.index[index], 'VKORC1 rs9923231'] = 'A/A'
                if row['VKORC1 rs8050894'] == 'C/G':
                    df.loc[df.index[index], 'VKORC1 rs9923231'] = 'A/G'
            if row['VKORC1 rs9934438'] == 'C/C':
                df.loc[df.index[index], 'VKORC1 rs9923231'] = 'G/G'
            if row['VKORC1 rs9934438'] == 'T/T':
                df.loc[df.index[index], 'VKORC1 rs9923231'] = 'A/A'
            if row['VKORC1 rs9934438'] == 'C/T':
                df.loc[df.index[index], 'VKORC1 rs9923231'] = 'A/G'


    # --- label binarizer for VKORC1 ---
    # only keep columns A/G, A/A, unknown (there are 126 missing still)
    def vkorc1_ag(s):
        if s['VKORC1 rs9923231'] == 'A/G':
            return 1
        else:
            return 0
    df['VKORC1 A/G'] = df.apply(vkorc1_ag, axis=1)

    def vkorc1_aa(s):
        if s['VKORC1 rs9923231'] == 'A/A':
            return 1
        else:
            return 0
    df['VKORC1 A/A'] = df.apply(vkorc1_aa, axis=1)

    def vkorc1_na(s):
        if pd.isnull(s['VKORC1 rs9923231']):
            return 1
        else:
            return 0
    df['VKORC1 Missing'] = df.apply(vkorc1_na, axis=1)

    # ---- DELETE MISSING ENTRIES THAT CANNOT BE IMPUTED
    # ---- AGE ----
    # discard all entries with missing age (42)
    df = df[df['Age'].notna()]

    # ----- HEIGHT AND WEIGHT -----
    # Drop all missing values of height and weight
    df = df[df['Height (cm)'].notna()]
    df = df[df['Weight (kg)'].notna()] # 4490

    # TREATMENT
    df = df[df['Therapeutic Dose of Warfarin'].notna()] # 4386


    ## ------ RACE ------
    lb = LabelBinarizer()
    race = lb.fit_transform(df['Race (OMB)'])
    classes = lb.classes_
    race = pd.DataFrame(race, columns=classes)
    # only keep Asian, Black/African American, Unknown
    ## --- POTENTIAL PROBLEM: most of the people are white --> missing group
    race = race[['Asian', 'Black or African American', 'Unknown']].rename(columns={'Unknown': 'Unknown Race'}) # must combine this with full dataset

    # --- cyp2C9 genotype ---
    #print(df['Cyp2C9 genotypes'].value_counts())
    #print(df['Cyp2C9 genotypes'].isna().sum())
    df['Cyp2C9 genotypes'] = df['Cyp2C9 genotypes'].fillna('Unknown Cyp2C9')
    lb = LabelBinarizer()
    cyp2c9 = lb.fit_transform(df['Cyp2C9 genotypes'])
    classes = lb.classes_
    cyp2c9 = pd.DataFrame(cyp2c9, columns=classes)
    cyp2c9 = cyp2c9[['*1/*1', '*1/*3', '*2/*2', '*2/*3', '*3/*3', 'Unknown Cyp2C9']] # need to add this back to all dataframe

    # --- ENZYME INDUCER ---
    # 1 if patient taking carbamazepine (tegretol), phenytoin (dilantin), rifampin or rifampicin, otherwise zero
    def enzyme(s):
        if (s['Carbamazepine (Tegretol)'] == 1) or (s['Phenytoin (Dilantin)'] == 1) or (s['Rifampin or Rifampicin'] == 1):
            return 1
        else:
            return 0

    df['Enzyme Inducer'] = df.apply(enzyme, axis=1)
    #print(df['Enzyme Inducer'].value_counts())

    # --- AMIODARONE STATUS ---
    # 1 if patient taking amiodarone (cordarone), otherwise zero
    # there are 1518 nan values
    df['Amiodarone (Cordarone)'] = df['Amiodarone (Cordarone)'].fillna(0)
    #print(df['Amiodarone (Cordarone)'].value_counts())


    def stratify(m, columns):
        for i in range(len(m)):
            if m[i][0] == 1:
                m[i] = [1, 1, 1, 1, 1]
            elif m[i][1] == 1:
                m[i] = [0, 1, 1, 1, 1]
            elif m[i][2] == 1:
                m[i] = [0, 0, 1, 1, 1]
            elif m[i][3] == 1:
                m[i] = [0, 0, 0, 1, 1]
            elif m[i][4] == 1:
                m[i] = [0, 0, 0, 0, 1]
        m = pd.DataFrame(data=m, columns=columns)
        return m

    # ---- DISCRETIZING AGE, HEIGHT, WEIGHT ----
    ### AGE
    df['Age'] = df['Age'].astype(str).str[0].astype(int)

    bins = [0, 2, 4, 6, 7, 9]
    labels = [1, 2, 3, 4, 5]
    df['age_bins'] = pd.cut(df['Age'], bins=bins, labels=labels)

    lb = LabelBinarizer()
    age = lb.fit_transform(df['age_bins'])
    age = stratify(age, ['Age1-2', 'Age3-4', 'Age5-6', 'Age7', 'Age8-9'])

    ### HEIGHT
    buffer = df['Height (cm)'].sort_values().to_numpy()
    inc = int(len(buffer)/5)
    bins = [buffer[0]-1, buffer[inc], buffer[inc*2], buffer[inc*3], buffer[inc*4], buffer[-1]]
    #bins = [124.97, 158.0, 165.1, 170.94, 178.0, 202.0]
    labels = [1, 2, 3, 4, 5]
    df['height_bins'] = pd.cut(df['Height (cm)'], bins=bins, labels=labels)

    lb = LabelBinarizer()
    height = lb.fit_transform(df['height_bins'])
    height = stratify(height, ['Height1', 'Height2', 'Height3', 'Height4', 'Height5'])

    ### WEIGHT
    buffer = df['Weight (kg)'].sort_values().to_numpy()
    inc = int(len(buffer)/5)
    bins = [buffer[0]-1, buffer[inc], buffer[inc*2], buffer[inc*3], buffer[inc*4], buffer[-1]]
    #bins = [124.97, 158.0, 165.1, 170.94, 178.0, 202.0]
    labels = [1, 2, 3, 4, 5]
    df['weight_bins'] = pd.cut(df['Weight (kg)'], bins=bins, labels=labels)

    lb = LabelBinarizer()
    weight = lb.fit_transform(df['weight_bins'])
    weight = stratify(weight, ['Weight1', 'Weight2', 'Weight3', 'Weight4', 'Weight5'])
    
    # combining everything together
    rest = df[['id', 'Gender', 'Enzyme Inducer', 'Amiodarone (Cordarone)', 'VKORC1 A/G', 'VKORC1 A/A', 'VKORC1 Missing', 'Therapeutic Dose of Warfarin']]
    rest = rest.reset_index().drop(columns=['index'])
    # COMPILE EVERYTHING (bucketized)

    # csv not bucketized
    rest2 = df[['Age', 'Height (cm)', 'Weight (kg)']]
    rest2 = rest2.rename(columns={'Height (cm)': 'Height', 'Weight (kg)': 'Weight'}).reset_index().drop(columns=['index'])
    df = pd.concat([rest2, race, cyp2c9, rest], axis=1)


    # --- DISCRETIZING OUTCOME AND TREATMENT ---
    # Drop all missing values of no optimal doe

    bins = [0, 21, 48, 350]
    labels = [0, 1, 2]

    # transform t_ideal from the formula
    # df['t_ideal'] = pd.cut(df['Therapeutic Dose of Warfarin'], bins=bins, labels=labels)

    cols = ['Age', 'Height', 'Weight', 'VKORC1 A/G', 'VKORC1 A/A', 'VKORC1 Missing', '*1/*1', '*1/*3', '*2/*2',
            '*2/*3', '*3/*3', 'Unknown Cyp2C9', 'Asian', 'Black or African American', 'Unknown Race',
            'Enzyme Inducer', 'Amiodarone (Cordarone)']

    coeffs = [-0.2614, 0.0087, 0.0128, -0.8677, -1.6974, -0.4854, -0.5211, -0.9357, -1.0616, -1.9206, -2.3312, -0.2188,
              -0.1092, -0.2760, -0.1032, 1.1816, -0.5503]

    const = 5.6044
    df['t_ideal'] = const
    for i in range(len(cols)):
        df['t_ideal'] += coeffs[i] * df[cols[i]]


    # Add noise
    noise = np.random.normal(0, 0.2, size=len(df))

    df['t_ideal_noise'] = df['t_ideal'] + noise

    df['t_ideal'] = df['t_ideal'] * df['t_ideal']
    df['t_ideal_noise'] = df['t_ideal_noise'] * df['t_ideal_noise']
    #print(skm.r2_score(df['Therapeutic Dose of Warfarin'], df['t_ideal']))
    df['t_ideal'] = pd.cut(df['t_ideal'], bins=bins, labels=labels)
    df['t_ideal_noise'] = pd.cut(df['t_ideal_noise'], bins=bins, labels=labels)

    print(df['t_ideal'].compare(df['t_ideal_noise'])) # ~248 changed their ideal treatment assignment

    df['y0'] = df['t_ideal_noise'].apply(lambda x: 1 if x == 0 else 0)
    df['y1'] = df['t_ideal_noise'].apply(lambda x: 1 if x == 1 else 0)
    df['y2'] = df['t_ideal_noise'].apply(lambda x: 1 if x == 2 else 0)

    # --- RANDOMLY GENERATE TRUE DATA (0.33)
    np.random.seed(seed)
    df['t'] = np.random.randint(3, size=len(df))
    l = []
    # take the outcome of y given the assigned treatment
    for index, row in df.iterrows():
        # take the assigned treatment
        t = row['t']
        l.append(row['y' + str(int(t))])
    df['y'] = l
    
    # NEW ADDITION -- add Gender and ID to assess disparity
    addl = df[['id', 'Gender']]
    
    rest = df[['Enzyme Inducer', 'Amiodarone (Cordarone)', 'VKORC1 A/G', 'VKORC1 A/A', 'VKORC1 Missing', 'y', 't', 'y0', 'y1', 'y2']]
    rest = rest.reset_index().drop(columns=['index'])
    # COMPILE EVERYTHING (bucketized)
    df_enc = pd.concat([addl, age, height, weight, race, cyp2c9, rest], axis=1)
    df_enc.to_csv(path + 'warfarin_enc_0.33.csv', index=False)

    # csv not bucketized
    rest2 = df[['Age', 'Height', 'Weight']]
    rest2 = rest2.reset_index().drop(columns=['index'])
    df_normal = pd.concat([addl, rest2, race, cyp2c9, rest], axis=1)

    df_normal.to_csv(path + 'warfarin_0.33.csv', index=False)


    def nonrandomized_v3(p, seed):
        np.random.seed(seed)

        new_coeffs = []
        for i in coeffs:
            lb = i - i * p
            ub = i + i * p
            new_coeffs.append(np.random.uniform(lb, ub))

        lb = const - const * p
        ub = const + const * p
        new_const = np.random.uniform(lb, ub)

        nonrandom = new_const
        for i in range(len(cols)):
            nonrandom += new_coeffs[i] * df[cols[i]]

        nonrandom = nonrandom * nonrandom
        nonrandom = pd.cut(nonrandom, bins=bins, labels=labels)
        # print(nonrandom.value_counts())
        # print(df['t_ideal'].value_counts())
        diff = df['t_ideal'].astype(int) - nonrandom.astype(int)
        diff_value = diff.value_counts()
        diff_value = diff_value[[i == 0 for i in diff_value.index]]
        num_correct = diff_value.sum()
        accuracy = float(num_correct)/4386

        print("SEED: " + str(seed))

        df['t'] = nonrandom
        l = []
        # take the outcome of y given the assigned treatment
        for index, row in df.iterrows():
            # take the assigned treatment
            t = row['t']
            l.append(row['y' + str(int(t))])
        df['y'] = l

        for t in [0, 1, 2]:
            print(t)
            df_buffer = df[df['t'] == int(t)]
            print(df_buffer['y'].value_counts())

        rest = df[
            ['Enzyme Inducer', 'Amiodarone (Cordarone)', 'VKORC1 A/G', 'VKORC1 A/A', 'VKORC1 Missing', 'y', 't', 'y0', 'y1',
             'y2']]
        rest = rest.reset_index().drop(columns=['index'])
        # COMPILE EVERYTHING (bucketized)
        df_enc = pd.concat([addl, age, height, weight, race, cyp2c9, rest], axis=1)
        df_enc.to_csv(path + 'warfarin_enc_r' + str(p) + '.csv', index=False)

        # csv not bucketized
        rest2 = df[['Age', 'Height', 'Weight']]
        rest2 = rest2.reset_index().drop(columns=['index'])
        df_normal = pd.concat([addl, rest2, race, cyp2c9, rest], axis=1)
        df_normal.to_csv(path + 'warfarin_r' + str(p) + '.csv', index=False)


    #nonrandomized_v3(0.3, 5)
    nonrandomized_v3(0.06, seed)
    nonrandomized_v3(0.11, seed)

for i, seed in enumerate([13, 1, 5, 19, 109]):
    df = pd.read_csv('../raw.csv')
    di = {'male': 0, 'female': 1}
    df = df.replace({"Gender": di})
    df = df.reset_index()
    df = df.rename(columns={'index': 'id'})
    path = f'../processed/warfarin_intermediate/seed{i+1}/'

    main(df, seed, path)


