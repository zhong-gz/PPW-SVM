import sys
sys.path.insert(0, sys.path[0]+"/../") # add parent directory to path
import numpy as np
from functions import accuracy,data_distribution_map1,data_distribution_map2,data_distribution_map3,\
    preprocess_data_shift,linear_data_generation,non_linear_data_generation
from datetime import datetime
import random
from two_stage_approach import two_stage_algo

def TSA(X,y,num_iters = 25,d_list = [10,1000,10000],map = 1,strat_features = np.array([1, 6, 8])-1,\
        num_experiments = 10,seed_value = 42):
    
    method_name = 'Two-Stage Approach'
    num_d  = len(d_list)
    n = X.shape[0]
    d = X.shape[1]

    print('Outside the echo chamber (two stage approach):')
    model_int = two_stage_algo(X_base = X,y_base = y)
    model_int.train(X, y)
    model_gaps         = np.zeros((num_experiments, num_d, num_iters)) #[[[] for _ in range(num_d)] for _ in range(num_experiments)]
    acc_list_start     = np.zeros((num_experiments, num_d, num_iters)) #[[[] for _ in range(num_d)] for _ in range(num_experiments)]
    acc_list_end       = np.zeros((num_experiments, num_d, num_iters)) #[[[] for _ in range(num_d)] for _ in range(num_experiments)]

    for i in range(num_experiments):
        print('  Running {} th times'.format(i))
        np.random.seed(seed_value + i)
        random.seed(seed_value  + i)
        for k, d in enumerate(d_list):
            # initial model
            print('    Running epsilon =  {}'.format(d))
            model = two_stage_algo(X_base = X,y_base = y)
            model.train(X, y)
            theta = np.copy(model.theta)
            
            for t in range(num_iters):
                # adjust distribution to current theta
                if map == 1:
                    X,y = linear_data_generation(n = n)
                    X_strat,y_strat = data_distribution_map1(X, y,mu = d, model = model, strat_features = strat_features)
                    # X_strat = preprocess_data_shift(X_strat, X, strat_features, n)
                
                if map == 2:
                    X_strat,y_strat = data_distribution_map2(X, y,mu = d, model = model)
                
                if map == 3:
                    X_strat,y_strat = data_distribution_map3(X, y,mu = d, model = model, strat_features = strat_features,t = t)
                    X_strat = preprocess_data_shift(X_strat, X, strat_features, n)

                if map == 4:
                    X,y = non_linear_data_generation(n = n)
                    X_strat,y_strat = data_distribution_map1(X, y,mu = d, model = model, strat_features = strat_features)
                    # X_strat = preprocess_data_shift(X_strat, X, strat_features, n)
                if map == 5:
                    X,y = linear_data_generation(n = n)
                    X_strat,y_strat = data_distribution_map2(X, y,mu = d, model = model)     
                if map == 6:
                    X,y = non_linear_data_generation(n = n)
                    X_strat,y_strat = data_distribution_map2(X, y,mu = d, model = model)
                # evaluate initial loss on the current distribution
                _,pred_label = model.predict(X_strat)
                acc = accuracy(y_strat, pred_label)
                acc_list_start[i,k,t] = acc
                
                # # learn on induced distribution
                model.train(X_strat, y_strat)
                theta_new = np.copy(model.theta)
                if np.isnan(np.dot(theta_new,theta.T)/(np.linalg.norm(theta_new)*np.linalg.norm(theta))):
                    model_gaps[i,k,t] = 0
                else:
                    model_gaps[i,k,t] = np.dot(theta_new,theta.T)/(np.linalg.norm(theta_new)*np.linalg.norm(theta))
                # model_gaps[i,k,t] = np.linalg.norm(theta_new - theta)
                theta = np.copy(theta_new)

                # evaluate final loss on the current distribution
                _,pred_label_new = model.predict(X_strat)
                acc = accuracy(y_strat, pred_label_new)
                acc_list_end[i,k,t] = acc
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
    print("Completion Time:", current_time_str)

    return model_gaps_avg,model_gaps_std,acc_list_start_avg,acc_list_start_std,acc_list_end_avg,acc_list_end_std,method_name
