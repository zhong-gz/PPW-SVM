import pandas as pd
import numpy as np
from sklearn import preprocessing
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.under_sampling import RandomUnderSampler,NearMiss


def load_data(file_loc):

    data = pd.read_csv(file_loc)

    y=data['Bankrupt?']
    X=data.drop(['Bankrupt?'],axis=1)

    nm1 = NearMiss(version=3)
    X, y = nm1.fit_resample(X, y)
    # rus = RandomUnderSampler(random_state=42)
    # X, y = rus.fit_resample(X, y)

    X = X.values
    X = X.astype(float)
    Y_all = y.values
    y = Y_all.astype(float)
    y[y != 1] = -1

    # balance classes
    default_indices = np.where(y == 1)[0]
    other_indices = np.where(y == -1)[0]
    indices = np.concatenate((default_indices, other_indices))

    X_balanced = X[indices]
    Y_balanced = y[indices]

    # shuffle arrays
    p = np.random.permutation(len(indices))
    X_full = X_balanced[p]
    Y_full = Y_balanced[p]
    
    # zero mean, unit variance
    X_full = preprocessing.scale(X_full)

    return X_full, Y_full, data
    # return X_full, Y_full, data
