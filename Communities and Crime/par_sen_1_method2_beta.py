import sys
sys.path.insert(0, sys.path[0]+"/../") # add parent directory to path
sys.path.insert(0, sys.path[0]+"Algorithm")
import numpy as np
import pandas as pd
import random
from Algorithm.alg_method_1 import method_1
from Algorithm.alg_method_2 import method_2
from Algorithm.plot_par_sen import plot_fig_par
from functions import linear_data_generation,non_linear_data_generation

# problems parameters
threshold = 0.1
seed_value = 42
num_iters = 50
d_list = [5,7.5,10]
num_experiments = 10
map = 2
folder_path = 'result_par_sen_1/method2_beta/'

initial=pd.read_csv('communities-crime-clean.csv')
initial = initial.drop('communityname', axis=1)
initial = initial.drop('fold', axis=1)
initial = initial.drop('state', axis=1)
y = np.where(initial['ViolentCrimesPerPop']>threshold, 1, -1)
initial = initial.drop('ViolentCrimesPerPop', axis=1)
X = initial.values 
pos=initial[(y== 1)]
pos_percentage=len(pos)/len(initial)
neg_percentage=1-pos_percentage
print('positive instance percentage is ',pos_percentage)
print('negative instance percentage is ',neg_percentage)

n = X.shape[0]
d = X.shape[1]
print('Sample number : ',n)
print('Sample dimension : ',d)
kerneltype = 'linear'
s = 1
betas = [0.1,0.3,0.5,0.7,0.9]
latex_text = r"$\beta$"

for beta in betas:
    print('beta = ',beta)
    # method 2
    model_gaps_avg,model_gaps_std,acc_list_start_avg,acc_list_start_std,acc_list_end_avg,acc_list_end_std,method_name = \
        method_2(X,y,num_iters,d_list,map = map,kerneltype = kerneltype,s =s,\
                 num_experiments = num_experiments,seed_value = seed_value,beta = beta)
    file_name_npy = f"{folder_path}{beta}.npz"
    np.savez(file_name_npy, model_gaps_avg = model_gaps_avg, model_gaps_std = model_gaps_std,\
                acc_list_start_avg = acc_list_start_avg, acc_list_start_std = acc_list_start_std,\
                acc_list_end_avg = acc_list_end_avg, acc_list_end_std = acc_list_end_std)
    print(f"Data saved to {file_name_npy}")

plot_fig_par(num_iters,d_list,folder_path,betas,latex_text)