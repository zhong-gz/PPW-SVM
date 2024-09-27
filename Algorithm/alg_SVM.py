import sys
sys.path.insert(0, sys.path[0]+"/../") # add parent directory to path
import numpy as np
from svm_self import svm_no_b
from functions import accuracy,est_varepsilon,data_distribution_map1,data_distribution_map2,data_distribution_map3,\
    remove_outliers_iqr,preprocess_data_shift,linear_data_generation,non_linear_data_generation
import copy
from datetime import datetime
import random
import time
import kernel

def SVM(X,y,num_iters = 25,d_list = [10,1000,10000],map = 1,kerneltype = 'linear',s = 0.05,\
             strat_features = None,num_experiments = 10,seed_value = 42):
    
    # random_norm = np.linalg.norm(random_data,axis=1)

    method_name = 'SVM_with_fix_C'
    num_d  = len(d_list)

    n = X.shape[0]
    m = X.shape[1]
    C = 100*n
    svm_int = svm_no_b(C=C, kernelType=kerneltype,gamma = s)
    svm_int.train(X, y)
    print('SVM with fix C:')
    model_list         = [[[svm_int] for _ in range(num_d)] for _ in range(num_experiments)]
    model_gaps         = np.zeros((num_experiments, num_d, num_iters)) #[[[] for _ in range(num_d)] for _ in range(num_experiments)]
    acc_list_start     = np.zeros((num_experiments, num_d, num_iters)) #[[[] for _ in range(num_d)] for _ in range(num_experiments)]
    acc_list_end       = np.zeros((num_experiments, num_d, num_iters)) #[[[] for _ in range(num_d)] for _ in range(num_experiments)]

    for i in range(num_experiments):
        print('  Running {} th times'.format(i))
        np.random.seed(seed_value + i)
        random.seed(seed_value  + i)

        for k, d in enumerate(d_list):
            # initial model
            svm = copy.deepcopy(svm_int)

            print('     Running epsilon =  {}'.format(d))
            
            for t in range(num_iters):
                print(f'       Current iteration =  {t+1}, there are still {num_iters - t -1} iterations left', end='\r')
                # adjust distribution to current theta
                if map == 1:
                    X,y = linear_data_generation(n = n)
                    X_strat,y_strat = data_distribution_map1(X, y,mu = d, model = svm, strat_features = strat_features)
                    # X_strat = preprocess_data_shift(X_strat, X, strat_features, n)
                    
                if map == 2:
                    X_strat,y_strat = data_distribution_map2(X, y,mu = d, model = svm)

                if map == 3:
                    X_strat,y_strat = data_distribution_map3(X, y,mu = d, model = svm, strat_features = strat_features,t = t)
                    X_strat = preprocess_data_shift(X_strat, X, strat_features, n)

                if map == 4:
                    X,y = non_linear_data_generation(n = n)
                    X_strat,y_strat = data_distribution_map1(X, y,mu = d, model = svm, strat_features = strat_features)
                    # X_strat = preprocess_data_shift(X_strat, X, strat_features, n)
                    
                # evaluate initial loss on the current distribution
                _,pred_label = svm.predict(X_strat)
                acc = accuracy(y_strat, pred_label)
                acc_list_start[i,k,t] = acc

                # learn on induced distribution
                svm_new = svm_no_b(C=C, kernelType=kerneltype,gamma = s)
                svm_new.train(X_strat, y_strat)

                # evaluate final loss on the current distribution
                _,pred_label_new = svm_new.predict(X_strat)
                acc = accuracy(y_strat, pred_label_new)
                acc_list_end[i,k,t] = acc

                # keep track of statistics
                model_list[i][k].append(svm_new)

                w_t = model_list[i][k][-1]
                w_t_1 = model_list[i][k][-2]
                temp  = kernel.compute_kernel_matrix(w_t.support_vector,w_t_1.support_vector,kernel=w_t_1.kernelType,parameter=w_t_1.gamma) + min(w_t.R,w_t_1.R)
                temp1 = w_t.support_alpha * w_t.support_label
                temp2 = w_t_1.support_alpha * w_t_1.support_label
                w_t_w_t_1 = np.dot(np.dot(temp1.T,temp),temp2)
                model_gaps[i,k,t] = w_t_w_t_1/(w_t.norm() * w_t_1.norm())

                svm = copy.deepcopy(svm_new)
            print('')
            # print('       C = ',C,'\n')
        print('-'*50)

    for k, d in enumerate(d_list):
        model_gaps_avg = np.mean(model_gaps, axis=0)
        model_gaps_std = np.std(model_gaps, axis=0)
        acc_list_start_avg = np.mean(acc_list_start, axis=0)
        acc_list_start_std = np.std(acc_list_start, axis=0)
        acc_list_end_avg = np.mean(acc_list_end, axis=0)
        acc_list_end_std = np.std(acc_list_end, axis=0)

    current_time = datetime.now()
    current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print(method_name," Completion Time: ", current_time_str)

    return model_gaps_avg,model_gaps_std,acc_list_start_avg,acc_list_start_std,acc_list_end_avg,acc_list_end_std,method_name
