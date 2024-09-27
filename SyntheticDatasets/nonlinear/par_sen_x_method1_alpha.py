import sys
sys.path.insert(0, sys.path[0]+"/../") # add parent directory to path
sys.path.insert(0, sys.path[0]+"/../")
sys.path.insert(0, sys.path[0]+"Algorithm")
import numpy as np
import random
from Algorithm.alg_method_1 import method_1
from Algorithm.plot_par_sen import plot_fig_par
from functions import linear_data_generation,non_linear_data_generation

# problems parameters
seed_value = 42
num_iters = 50
d_list = [0.1,0.3,0.5,0.7] #0.1,0.3,0.5,0.7
num_experiments = 10
map = 4
folder_path = 'result_par_sen_x/method1/'

np.random.seed(seed_value)
random.seed(seed_value)
n = 200
X,y = non_linear_data_generation(n = n)
d = X.shape[1]
print('Sample number : ',n)
print('Sample dimension : ',d)
print('-'*50)
strat_features = None
kerneltype = 'rbf'
s = 1
alphas = [0.1, 0.2, 0.3, 0.4, 0.49]

for alpha in alphas:
    print('alpha = ',alpha)
    # method 1
    model_gaps_avg,model_gaps_std,acc_list_start_avg,acc_list_start_std,acc_list_end_avg,acc_list_end_std,method_name = \
        method_1(X,y,num_iters,d_list,map = map,kerneltype = kerneltype,s = s,strat_features = strat_features,\
                 num_experiments = num_experiments,seed_value = seed_value,alpha = alpha)
    file_name_npy = f"{folder_path}{alpha}.npz"
    np.savez(file_name_npy, model_gaps_avg = model_gaps_avg, model_gaps_std = model_gaps_std,\
                acc_list_start_avg = acc_list_start_avg, acc_list_start_std = acc_list_start_std,\
                acc_list_end_avg = acc_list_end_avg, acc_list_end_std = acc_list_end_std)
    print(f"Data saved to {file_name_npy}")

plot_fig_par(num_iters,d_list,folder_path,alphas)