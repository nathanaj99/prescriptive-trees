import pandas as pd
import math
from sklearn.preprocessing import LabelBinarizer
import numpy as np

df = pd.read_csv('../../Warfarin/raw.csv')

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




# --- DISCRETIZING OUTCOME AND TREATMENT ---
# Drop all missing values of no optimal doe

bins = [0, 21, 48, 350]
labels = [0, 1, 2]
df['t_ideal'] = pd.cut(df['Therapeutic Dose of Warfarin'], bins=bins, labels=labels)

# construct "true outcomes"
# 1 if coincides with treatment tier
df['y0'] = df['t_ideal'].apply(lambda x: 1 if x == 0 else 0)
df['y1'] = df['t_ideal'].apply(lambda x: 1 if x == 1 else 0)
df['y2'] = df['t_ideal'].apply(lambda x: 1 if x == 2 else 0)

# --- RANDOMLY GENERATE TRUE DATA (0.5)

df['t'] = np.random.randint(3, size=len(df))
l = []
# take the outcome of y given the assigned treatment
for index, row in df.iterrows():
    # take the assigned treatment
    t = row['t']
    l.append(row['y' + str(int(t))])
df['y'] = l

rest = df[['VKORC1 A/G', 'VKORC1 A/A','VKORC1 Missing', 'y', 't', 'y0', 'y1', 'y2']]
rest = rest.reset_index().drop(columns=['index'])
# COMPILE EVERYTHING (bucketized)
dataframe = pd.concat([age, height, weight, race, cyp2c9, rest], axis=1)
dataframe.to_csv('warfarin_enc.csv', index=False)

# csv not bucketized
rest2 = df[['Age', 'Height (cm)', 'Weight (kg)']]
rest2 = rest2.rename(columns={'Height (cm)': 'Height', 'Weight (kg)': 'Weight'}).reset_index().drop(columns=['index'])
df_enc = pd.concat([rest2, race, cyp2c9, rest], axis=1)
df_enc.to_csv('warfarin.csv', index=False)

"""print(df['Height (cm)'].describe())
print(df['Height (cm)'].isna().sum()) # 1146 missing
print(df['Weight (kg)'].describe())
print(df['Weight (kg)'].isna().sum()) # 287 missing
print(df['Age'].value_counts())
print(df['Age'].isna().sum()) # 42 missing

weight = df['Weight (kg)'].isna()
height = df['Height (cm)'].isna()
age = df['Age'].isna()
num = 0
num_weight = 0
num_height = 0
for i in range(len(weight)):
    if weight[i] and height[i]:
        num += 1
    elif weight[i] and not height[i]:
        num_weight += 1
    elif height[i] and not weight[i]:
        num_height += 1

print(num_weight) # 31
print(num_height) # 890"""


# if weight is available but not height (31) impute
# average US male BMI is 28.6, female is 28.7
# height (cm) = sqrt(mass (kg) / BMI) * 100

# age < 10: 16.5
# age 10-20: 25 male, 24 female
# age 20-29: male 26.8, female 27.5
# age 30-60: male BMI is 28.6, female is 28.7
# age 61-69: male 29.5, female 29.6
# age 70+: 26

# if height is available but not weight (890) impute
# weight (kg) = (height (cm) / 100)^2 * BMI
