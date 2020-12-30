import pandas as pd
import sklearn.linear_model as lm
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import normalize
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import sklearn.metrics as skm
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from itertools import combinations
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from sklearn.linear_model import LogisticRegression
from collections import Counter


pd.set_option('display.max_columns', None)

def preprocessing():
    df = pd.read_csv('../../IST Data/IST_train.csv')
    t_cols = ['DASP14', 'DLH14', 'DMH14']
    """treatments = df[t_cols]
    print(treatments)"""

    print(df['DIED'].value_counts())

    df['t'] = df['DASP14'] + df['DLH14'] + df['DMH14']

    treat_map = {'YYN': 1, 'YNY': 2, 'YNN': 3, 'NYN': 4, 'NNY': 5, 'NNN': 6}
    df['t'] = df['t'].map(treat_map)#.astype(int)

    # there are 53 entries with NA, because of NaN or unknown 'U' data. Just drop it
    df = df[~df['t'].isna()]
    df['t'] = df['t'].astype(int)

    binary_map = {'Y': 1, 'N': 0, 'U': 0}

    y_cols = ['FRECOVER', 'DALIVE', 'FDENNIS', 'DRSISC', 'DRSH', 'DRSUNK', 'DPE', 'DSIDE']

    for i in y_cols:
        df[i] = df[i].map(binary_map)

    def recurrent(r):
        if r.DRSISC == 1 or r.DRSH == 1 or r.DRSUNK == 1:
            return 1
        else:
            return 0

    df['recurrent'] = df.apply(recurrent, axis=1)

    """
    DESCRIPTION OF NaN in y_cols:
    - FRECOVER and FDENNIS: NaN is they died (NaN -> 00
    - DRSUNK: No data for pilot project (NaN -> 0)
    - DALIVE: Simply no data (NaN -> 0)
    """
    df['FRECOVER'] = df['FRECOVER'].fillna(0)
    df['FDENNIS'] = df['FDENNIS'].fillna(0)
    df['DRSUNK'] = df['DRSUNK'].fillna(0)
    df['DALIVE'] = df['DALIVE'].fillna(0)

    df['y'] = 2 * df['FRECOVER'] + df['DALIVE'] - 2 * df['DIED'] - df['FDENNIS'] - df['recurrent'] - 0.5 * df['DPE'] - 0.3 * df['DSIDE']
    maximum = max(df['y'])
    minimum = min(df['y'])
    df['y'] = (df['y'] - minimum)/(maximum - minimum)

    X = df.iloc[:, :14].reset_index().drop(columns=['index'])
    #norm = Normalizer()
    y = df['y'].reset_index().drop(columns=['index'])
    #print(y)
    t = df['t'].reset_index().drop(columns=['index'])


    ## BINARIZE MULTI-LEVEL VARIABLES (RCONSC, STYPE)
    lb = LabelBinarizer()
    rconsc = lb.fit_transform(X['RCONSC'])
    stype = lb.fit_transform(X['STYPE'])
    rconsc = pd.DataFrame(data=rconsc, columns=['RCONSC1', 'RCONSC2', 'RCONSC3'])
    stype = pd.DataFrame(data=stype, columns=['STYPE1', 'STYPE2', 'STYPE3', 'STYPE4', 'STYPE5'])

    # Binarize SEX
    sex_map = {'M': 1, 'F': 0}
    X['SEX'] = X['SEX'].map(sex_map)

    # Binarize RVISINF and RDEF1-8
    rdef_map = {'Y': 1, 'N': 0, 'C': 0}
    X['RVISINF'] = X['RVISINF'].map(rdef_map)
    for i in range(1, 9):
        col_name = 'RDEF' + str(i)
        X[col_name] = X[col_name].map(rdef_map)

    X = X.drop(columns=['RCONSC', 'STYPE'])
    X = pd.concat([X, rconsc, stype], axis=1)

    df = pd.concat([X, y, t], axis=1)

    df.to_csv('IST_buffer.csv', index=False)



def learn():
    df = pd.read_csv('IST_buffer.csv')

    t_unique = df['t'].unique()
    test = {}
    errors = {}

    for i in t_unique:
        buffer = df[df['t'] == i]
        X = buffer.iloc[:, :20]
        y = buffer['y']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Apply SMOTE to only training data
        #smote = SMOTE(sampling_strategy=1.0)
        #X_train, y_train = smote.fit_resample(X_train, y_train)

        # ADASYN is promising
        # KMeans SMOTE

        #parameters = {'max_depth': range(5, 30)}
        #clf = GridSearchCV(RandomForestClassifier(), parameters, n_jobs=-1, cv=5, scoring='accuracy')
        #C = {'C': np.logspace(-4, 4, 20)}
        #clf = GridSearchCV(LogisticRegression(max_iter=1000, solver='saga'), C, n_jobs=-1, cv=5, scoring='accuracy')

        parameters = {'max_depth': range(5, 30)}
        clf = GridSearchCV(RandomForestRegressor(), parameters, n_jobs=-1, cv=5, scoring='neg_mean_squared_error')
        clf.fit(X_train, y_train)
        lr = clf.best_estimator_
        #print(clf.best_score_, clf.best_params_)
        print(lr.score(X_test, y_test))
        test_predict = lr.predict(X_test)


        error = y_test - test_predict
        mu, std = norm.fit(error)
        print(mu, std)

        test[i] = lr
        errors[i] = [mu, std]
        """print(error)
        plt.hist(error)
        plt.show()"""


        """tn, fp, fn, tp = skm.confusion_matrix(y_test, test_predict).ravel()
        #print(tn, fp, fn, tp)
        tpr = tp / float(tp + fn)
        tnr = tn / float(tn + fp)
        print(tpr, tnr)

        test[i] = lr
        errors[i] = [tpr, tnr]"""


    """lr2 = lm.LassoCV().fit(X, y)
    print(lr2.score(X, y))"""
    #print(lr.get_depth())
    #lr = lm.LassoCV().fit(X, y)
    #lr = lm.LinearRegression().fit(X, y)




    # Predict on current training set, and find the respective errors
    # visualize errors on train and test set (split first)
    #pred_y = lr.predict(X)



    for i in range(1, 7):
        model = test[i]
        X = df.iloc[:, :20]
        prediction = model.predict(X)

        """for idx, pred in enumerate(prediction):
            if pred == 1:
                error = np.random.binomial(1, errors[i][0]) # tpr
            else:
                error = np.random.binomial(1, errors[i][1]) #tnr
            #print(pred)
    
            # if error 1, don't change, if 0, then change
            if error == 0:
                pred ^= 1
                prediction[idx] = pred
            #print(error, pred)"""

        error = np.random.normal(errors[i][0], errors[i][1], len(X))
        prediction += error

        maximum = max(prediction)
        minimum = min(prediction)
        prediction = (prediction - minimum) / (maximum - minimum)

        df['y' + str(i)] = prediction

    #df = df.rename(columns={'DIED':'y'})
    df.to_csv('cleaned_IST.csv', index=False)



def correl_analysis():
    y_cols = ['y1', 'y2', 'y3', 'y4', 'y5', 'y6']
    df = pd.read_csv('cleaned_IST_noerror.csv')

    # this is here because i forgot to normalize in the previous function
    for i in y_cols:
        maximum = max(df[i])
        minimum = min(df[i])
        df[i] = (df[i] - minimum) / (maximum - minimum)

    df.to_csv('cleaned_IST.csv', index=False)

    subset = df[y_cols]
    print(subset.corr())

def take_y():
    for i in range(1, 6):
        l = []
        name = '../data/IST/data_test_' + str(i) + '.csv'
        df = pd.read_csv(name)
        for index, row in df.iterrows():
            # take the assigned treatment
            t = row['t']
            l.append(row['y' + str(int(t))])

        df['y'] = l

        df.to_csv(name, index=False)

    for i in range(1, 6):
        l = []
        name = '../data/IST/data_train_' + str(i) + '.csv'
        df = pd.read_csv(name)
        for index, row in df.iterrows():
            # take the assigned treatment
            t = row['t']
            l.append(row['y' + str(int(t))])

        df['y'] = l

        df.to_csv(name, index=False)

def recalibrate():
    file_path = '../data/IST/'
    #m = {'y1': 'y0', 'y2': 'y1', 'y3': 'y2', 'y4': 'y3', 'y5': 'y4', 'y6': 'y5', 'ml1': 'ml0', 'ml2': 'ml1',
         #'ml3': 'ml2', 'ml4': 'ml3', 'ml5': 'ml4', 'ml6': 'ml5'}
    for i in range(1, 6):
        file_name = 'data_train_' + str(i) + '.csv'
        df = pd.read_csv(file_path + file_name)
        df['t'] = df['t'] - 1
        #df = df.rename(columns=m)
        df.to_csv(file_path + file_name, index=False)

    for i in range(1, 6):
        file_name = 'data_test_' + str(i) + '.csv'
        df = pd.read_csv(file_path + file_name)
        df['t'] = df['t'] - 1
       # df = df.rename(columns=m)
        df.to_csv(file_path + file_name, index=False)

def discretize():
    # need to discretize age and blood pressure
    file_dir = '../data/IST/'
    for i in range(1, 6):
        file_name = 'data_train_' + str(i) + '.csv'
        df = pd.read_csv(file_dir + file_name)

        X = df.iloc[:, :20]
        rest = df.iloc[:, 20:]

        ## DISCRETIZE AGE
        bins = [0, 25, 64, 100]
        labels = [1, 2, 3]
        df['age_bins'] = pd.cut(df['AGE'], bins=bins, labels=labels)

        lb = LabelBinarizer()
        age = lb.fit_transform(df['age_bins'])
        age = pd.DataFrame(data=age, columns=['AGE1', 'AGE2', 'AGE3'])

        #print(df[['AGE', 'age_bins']])

        bins = [0, 120, 139, 159, 300]
        labels = [1, 2, 3, 4]
        df['rsbp_bins'] = pd.cut(df['RSBP'], bins=bins, labels=labels)
        rsbp = lb.fit_transform(df['rsbp_bins'])
        rsbp = pd.DataFrame(data=rsbp, columns=['RSBP1', 'RSBP2', 'RSBP3', 'RSBP4'])

        df = pd.concat([X, age, rsbp, rest], axis=1)
        df = df.drop(columns=['AGE', 'RSBP'])

        df.to_csv(file_dir + 'data_train_enc_' + str(i) + '.csv', index=False)

    for i in range(1, 6):
        file_name = 'data_test_' + str(i) + '.csv'
        df = pd.read_csv(file_dir + file_name)

        X = df.iloc[:, :20]
        rest = df.iloc[:, 20:]

        ## DISCRETIZE AGE
        bins = [0, 25, 64, 100]
        labels = [1, 2, 3]
        df['age_bins'] = pd.cut(df['AGE'], bins=bins, labels=labels)

        lb = LabelBinarizer()
        age = lb.fit_transform(df['age_bins'])
        age = pd.DataFrame(data=age, columns=['AGE1', 'AGE2', 'AGE3'])

        # print(df[['AGE', 'age_bins']])

        bins = [0, 120, 139, 159, 300]
        labels = [1, 2, 3, 4]
        df['rsbp_bins'] = pd.cut(df['RSBP'], bins=bins, labels=labels)
        rsbp = lb.fit_transform(df['rsbp_bins'])
        rsbp = pd.DataFrame(data=rsbp, columns=['RSBP1', 'RSBP2', 'RSBP3', 'RSBP4'])

        df = pd.concat([X, age, rsbp, rest], axis=1)
        df = df.drop(columns=['AGE', 'RSBP'])

        df.to_csv(file_dir + 'data_test_enc_' + str(i) + '.csv', index=False)

discretize()



#prediction += error
#df['y' + str(i)] = prediction

#print(df)

""" PROBLEMS TO SOLVE:
normalize y maybe?
train each treatment on a variety of models
- linear regression
- lasso/ridge regression
- decision tree regressor
- how to properly evaluate the error
    1. for each treatment, 
- visualize errors
    1. map the errors for each model--on training set (can't really map errors because the decision tree regressor does really well)
        perhaps we should try splitting the model and doing multiple tries
"""

#MODIFIED: Y = 2 * I[FRECOVER] (Y/N) + I[DALIVE] (Y/N) - 2I[DIED] (1/0) - I[FDENNIS] (Y/N) - I[DRSISC or DRSH or DRSUNK] (Y/N) - 0.5I[DPE or intracranial bleeding] (Y/N) - 0.3I[DSIDE] (Y/N)

