import sys
sys.path.insert(0, sys.path[0]+"/../") # add parent directory to path
sys.path.insert(0, sys.path[0]+"Algorithm")
import numpy as np
import random
from data_prep import load_data
from Algorithm.alg_method_1 import method_1
from Algorithm.alg_method_2 import method_2
from Algorithm.plot_par_sen import plot_fig_par
from functions import linear_data_generation,non_linear_data_generation
import warnings

warnings.filterwarnings("ignore")
# problems parameters
seed_value = 42
num_iters = 50
d_list = [100,250,500,750,1000] #,1000,10000 1,10,100,
num_experiments = 10
map = 3
np.random.seed(seed_value)
random.seed(seed_value)
X, y, data = load_data(r'data.csv')
n = X.shape[0]
d = X.shape[1]
print('Sample number : ',n)
print('Sample dimension : ',d)
strat_features = np.concatenate((np.arange(1, 15), np.arange(16, 19),np.arange(21, 78),np.arange(80, 83),\
                                 np.arange(84, 88),np.arange(89,90),np.arange(92,94),np.arange(95,96))) - 1 # for later indexing
# print('Performative Features:')
# for i, feature in enumerate(strat_features):
#     print(strat_features[i], data.columns[feature + 1])
print('-'*50)
kerneltype = 'rbf'
s = 0.1
folder_path = 'result_par_sen/method2_beta/'
betas = [0.1,0.3,0.5,0.7,0.9]
latex_text = r"$\beta$"

for beta in betas:
    print('beta = ',beta)
    # method 2
    model_gaps_avg,model_gaps_std,acc_list_start_avg,acc_list_start_std,acc_list_end_avg,acc_list_end_std,method_name = \
        method_2(X,y,num_iters,d_list,map = map,kerneltype = kerneltype,s =s,strat_features = strat_features,\
                 num_experiments = num_experiments,seed_value = seed_value,beta = beta)
    file_name_npy = f"{folder_path}{beta}.npz"
    np.savez(file_name_npy, model_gaps_avg = model_gaps_avg, model_gaps_std = model_gaps_std,\
                acc_list_start_avg = acc_list_start_avg, acc_list_start_std = acc_list_start_std,\
                acc_list_end_avg = acc_list_end_avg, acc_list_end_std = acc_list_end_std)
    print(f"Data saved to {file_name_npy}")

plot_fig_par(num_iters,d_list,folder_path,betas,latex_text)