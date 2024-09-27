import sys
sys.path.insert(0, sys.path[0]+"/../") # add parent directory to path
sys.path.insert(0, sys.path[0]+"/../")
import numpy as np
import itertools
from data_prep import load_data
from Algorithm.svm_self import svm_no_b
from functions import accuracy
from multiprocessing import Pool

np.random.seed(42) 

X, y, data = load_data(r'data.csv')
n = X.shape[0]
print('sample number : ',n)
kerneltype = 'rbf'

param_grid = {
    's': [1,0.5,0.1,0.05,0.01],
    'c1': [1],
}

best_params = {}
best_score = 0

for c1 in param_grid['c1']:
    for s in param_grid['s']:
        print('Running c1 = n *',c1,', s = ',s)
        kfolds = 5
        fold_scores = []
        for fold in range(kfolds):
            # print('         Running fold:', fold)
            fold_size = len(X) // kfolds
            indices = np.arange(len(X))
            valid_indices = indices[fold * fold_size : (fold + 1) * fold_size]
            train_indices = np.delete(indices, valid_indices)
            
            X_train, X_valid = X[train_indices], X[valid_indices]
            y_train, y_valid = y[train_indices], y[valid_indices]
            
            svm = svm_no_b(C=c1*n, kernelType=kerneltype,gamma = s)
            svm.train(X_train, y_train)

            _,y_pred = svm.predict(X_valid)
            
            fold_scores.append(accuracy(y_valid, y_pred))

        avg_score = np.mean(fold_scores)
        print('Accuracy of n*c1',{'c1': c1}, c1, 'and s', {'s':s},s,':',avg_score)
        print('-'*50)
        
        if avg_score > best_score:
            best_score = avg_score
            best_params = {'c1': c1,'s':s}

    print('Running c1 =  n*',c1,'linear kernel')
    kerneltype = 'linear'
    kfolds = 5
    fold_scores = []
    for fold in range(kfolds):
        # print('         Running fold:', fold)
        fold_size = len(X) // kfolds
        indices = np.arange(len(X))
        valid_indices = indices[fold * fold_size : (fold + 1) * fold_size]
        train_indices = np.delete(indices, valid_indices)
        
        X_train, X_valid = X[train_indices], X[valid_indices]
        y_train, y_valid = y[train_indices], y[valid_indices]
        
        svm = svm_no_b(C=c1*n, kernelType=kerneltype,gamma = s)
        svm.train(X_train, y_train)

        _,y_pred = svm.predict(X_valid)
        
        fold_scores.append(accuracy(y_valid, y_pred))

    avg_score = np.mean(fold_scores)
    print('Accuracy of n*c1',{'c1': c1}, c1, 'and linear kernel:',avg_score)
    print('-'*50)
    
    if avg_score > best_score:
        best_score = avg_score
        best_params = {'c1': c1,'s':'linear kernel'}

print('Best params:', best_params)
print('Best score:', best_score)
