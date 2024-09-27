import sys
sys.path.insert(0, sys.path[0]+"/../") # add parent directory to path
sys.path.insert(0, sys.path[0]+"/../")
sys.path.insert(0, sys.path[0]+"Algorithm")
import numpy as np
import random
from Algorithm.alg_method_1 import method_1
from Algorithm.alg_method_2 import method_2
from Algorithm.alg_RRM import RRM
from Algorithm.alg_RGD import RGD
from Algorithm.alg_PPNN import PPNN
from Algorithm.alg_Outside import TSA
from Algorithm.alg_PerGD import PerformativeGD
from Algorithm.alg_SVM import SVM
from Algorithm.plot import plot_fig
from functions import linear_data_generation

# problems parameters
seed_value = 42
num_iters = 100
d_list = [10,20,30,40,50] #,1000,10000 10,100,1000,60,80,100
num_experiments = 10
map = 1
folder_path = 'result_x/'

np.random.seed(seed_value)
random.seed(seed_value)
n = 200
X,y = linear_data_generation(n = n)
d = X.shape[1]
print('Sample number : ',n)
print('Sample dimension : ',d)
print('-'*50)
strat_features = None
kerneltype = 'linear'

# method 1
model_gaps_avg,model_gaps_std,acc_list_start_avg,acc_list_start_std,acc_list_end_avg,acc_list_end_std,method_name = \
    method_1(X,y,num_iters,d_list,map = map,kerneltype = kerneltype,strat_features = strat_features,num_experiments = num_experiments,seed_value = seed_value)
file_name_npy = f"{folder_path}{method_name}.npz"
np.savez(file_name_npy, model_gaps_avg = model_gaps_avg, model_gaps_std = model_gaps_std,\
            acc_list_start_avg = acc_list_start_avg, acc_list_start_std = acc_list_start_std,\
            acc_list_end_avg = acc_list_end_avg, acc_list_end_std = acc_list_end_std)
print(f"Data saved to {file_name_npy}")

# method 2
model_gaps_avg,model_gaps_std,acc_list_start_avg,acc_list_start_std,acc_list_end_avg,acc_list_end_std,method_name = \
    method_2(X,y,num_iters,d_list,map = map,kerneltype = kerneltype,strat_features = strat_features,num_experiments = num_experiments,seed_value = seed_value)
file_name_npy = f"{folder_path}{method_name}.npz"
np.savez(file_name_npy, model_gaps_avg = model_gaps_avg, model_gaps_std = model_gaps_std,\
            acc_list_start_avg = acc_list_start_avg, acc_list_start_std = acc_list_start_std,\
            acc_list_end_avg = acc_list_end_avg, acc_list_end_std = acc_list_end_std)
print(f"Data saved to {file_name_npy}")

# SVM
model_gaps_avg,model_gaps_std,acc_list_start_avg,acc_list_start_std,acc_list_end_avg,acc_list_end_std,method_name = \
    SVM(X,y,num_iters,d_list,map = map,kerneltype = kerneltype,strat_features = strat_features,num_experiments = num_experiments,seed_value = seed_value)
file_name_npy = f"{folder_path}{method_name}.npz"
np.savez(file_name_npy, model_gaps_avg = model_gaps_avg, model_gaps_std = model_gaps_std,\
            acc_list_start_avg = acc_list_start_avg, acc_list_start_std = acc_list_start_std,\
            acc_list_end_avg = acc_list_end_avg, acc_list_end_std = acc_list_end_std)
print(f"Data saved to {file_name_npy}")

# RRM
model_gaps_avg,model_gaps_std,acc_list_start_avg,acc_list_start_std,acc_list_end_avg,acc_list_end_std,method_name = \
    RRM(X,y,num_iters,d_list,map = map,strat_features = strat_features,num_experiments = num_experiments,seed_value = seed_value)
file_name_npy = f"{folder_path}{method_name}.npz"
np.savez(file_name_npy, model_gaps_avg = model_gaps_avg, model_gaps_std = model_gaps_std,\
            acc_list_start_avg = acc_list_start_avg, acc_list_start_std = acc_list_start_std,\
            acc_list_end_avg = acc_list_end_avg, acc_list_end_std = acc_list_end_std)
print(f"Data saved to {file_name_npy}")

# RGD
model_gaps_avg,model_gaps_std,acc_list_start_avg,acc_list_start_std,acc_list_end_avg,acc_list_end_std,method_name = \
    RGD(X,y,num_iters,d_list,map = map,strat_features = strat_features,num_experiments = num_experiments,seed_value = seed_value)
file_name_npy = f"{folder_path}{method_name}.npz"
np.savez(file_name_npy, model_gaps_avg = model_gaps_avg, model_gaps_std = model_gaps_std,\
            acc_list_start_avg = acc_list_start_avg, acc_list_start_std = acc_list_start_std,\
            acc_list_end_avg = acc_list_end_avg, acc_list_end_std = acc_list_end_std)
print(f"Data saved to {file_name_npy}")

# ppnn
model_gaps_avg,model_gaps_std,acc_list_start_avg,acc_list_start_std,acc_list_end_avg,acc_list_end_std,method_name = \
    PPNN(X,y,num_iters,d_list,map = map,strat_features = strat_features,num_experiments = num_experiments,seed_value = seed_value)
file_name_npy = f"{folder_path}{method_name}.npz"
np.savez(file_name_npy, model_gaps_avg = model_gaps_avg, model_gaps_std = model_gaps_std,\
            acc_list_start_avg = acc_list_start_avg, acc_list_start_std = acc_list_start_std,\
            acc_list_end_avg = acc_list_end_avg, acc_list_end_std = acc_list_end_std)
print(f"Data saved to {file_name_npy}")

# outside the echo chamber
model_gaps_avg,model_gaps_std,acc_list_start_avg,acc_list_start_std,acc_list_end_avg,acc_list_end_std,method_name = \
    TSA(X,y,num_iters,d_list,map = map,strat_features = strat_features,num_experiments = num_experiments,seed_value = seed_value)
file_name_npy = f"{folder_path}{method_name}.npz"
np.savez(file_name_npy, model_gaps_avg = model_gaps_avg, model_gaps_std = model_gaps_std,\
            acc_list_start_avg = acc_list_start_avg, acc_list_start_std = acc_list_start_std,\
            acc_list_end_avg = acc_list_end_avg, acc_list_end_std = acc_list_end_std)
print(f"Data saved to {file_name_npy}")

# PerformativeGD
model_gaps_avg,model_gaps_std,acc_list_start_avg,acc_list_start_std,acc_list_end_avg,acc_list_end_std,method_name = \
    PerformativeGD(X,y,num_iters,d_list,map = map,strat_features = strat_features,num_experiments = num_experiments,seed_value = seed_value)
file_name_npy = f"{folder_path}{method_name}.npz"
np.savez(file_name_npy, model_gaps_avg = model_gaps_avg, model_gaps_std = model_gaps_std,\
            acc_list_start_avg = acc_list_start_avg, acc_list_start_std = acc_list_start_std,\
            acc_list_end_avg = acc_list_end_avg, acc_list_end_std = acc_list_end_std)
print(f"Data saved to {file_name_npy}")

plot_fig(num_iters,d_list,folder_path)