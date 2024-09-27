import sys
sys.path.insert(0, sys.path[0]+"/../") # add parent directory to path
import os
import numpy as np
from functions import plot_model_gap,plot_step,plot_acc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datetime import datetime
import pandas as pd

# problems parameters
num_iters = 100
d_list = [10,1000,10000]

def plot_fig(num_iters = 25,d_list = [10,1000,10000],folder_path = 'result/'):
    num_d  = len(d_list)
    data_dict = {}
    # methods = ['method_1','RRM_Logistic_Regression']
    # methods = ['method_1']
    # methods = ['method_1', 'method_2','SVM_with_fix_C']
    # methods = ['method_1', 'method_2','RRM_Logistic_Regression','RGD_Logistic_Regression',\
    #            'Outside_the_echo_chamber','PerGD','Performative_Prediction_with_Neural_Network']
    methods = ['PPW-AVG', 'PPW-EMA','RRM Logistic Regression','RGD Logistic Regression',\
               'RRM with Neural Networks','Two-Stage Approach','PerGD']
    for methods_name in methods:
        data = np.load(folder_path+methods_name + '.npz')
    
        inner_dict = {key: data[key] for key in data.files}
        data_dict[methods_name] = inner_dict
    
        data.close()

    colors = ['b', 'g', 'r', 'c', 'm', 'k', 'y', 'orange','purple']
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'x','3']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':','-']

    for c in range(num_d):
        fig = plt.figure(figsize=(15,6))
        ax = fig.gca()
        offset = 0.8
        max_element = 0
        min_element = 100
        for i,(file_key, inner_dict) in enumerate(data_dict.items()):
            for j in range(1,num_iters):
                acc_list_start_avg = inner_dict.get('acc_list_start_avg')
                acc_list_end_avg = inner_dict.get('acc_list_end_avg')
                plot_step(j, offset, acc_list_start_avg, acc_list_end_avg, file_key,colors[i],markers[i],num_iters,c)
                arrays = [acc_list_start_avg,acc_list_end_avg]
                max_element = max(max_element,np.max(np.maximum.reduce(arrays)))
                min_element = min(min_element,np.min(np.minimum.reduce(arrays)))
        plt.xlabel('Iteration', fontsize = 18)
        plt.ylabel('Accuracy', fontsize = 18)
        plt.tick_params(labelsize=18)
        plt.ylim(0.3, 1)
        # plt.ylim(min(0.5,min_element), max_element)
        plt.legend(loc='lower right')
        # plt.title('Accuracy, d={}'.format(d_list[c]), fontsize = 18)
        file_name = f'acc_d = {d_list[c]}.pdf'
        plt.tight_layout()
        plt.savefig(folder_path+file_name, transparent=True, backend='pdf')
    
    for c in range(num_d):
        fig = plt.figure(figsize=(15,6))
        max_element = 0
        min_element = 100
        for i, methods_name in enumerate(methods):
            inner_dict = data_dict[methods_name]
            acc_list_start_avg = inner_dict.get('acc_list_start_avg')
            acc_list_start_std = inner_dict.get('acc_list_start_std')
            plot_acc(acc_list_start_avg[c],acc_list_start_std[c],colors[i],markers[i],linestyles[i],methods_name)
            max_element = max(max_element,np.max(np.maximum.reduce(acc_list_start_avg[c])))
            min_element = min(min_element,np.min(np.minimum.reduce(acc_list_start_avg[c])))
        plt.xlabel('Iteration', fontsize = 18)
        plt.ylabel('Accuracy', fontsize = 18)
        plt.tick_params(labelsize=18)
        plt.ylim(0.3, 1)
        # plt.ylim(min(0.5,min_element), max_element)
        plt.legend(loc='lower right')
        # plt.title('Accuracy after data distribution shift, d={}'.format(d_list[c]), fontsize = 18)
        file_name = f'acc_d = {d_list[c]}_start.pdf'
        plt.tight_layout()
        plt.savefig(folder_path+file_name, transparent=True, backend='pdf')

    for c in range(num_d):
        fig = plt.figure(figsize=(15,6))
        max_element = 0
        min_element = 100
        for i, methods_name in enumerate(methods):
            inner_dict = data_dict[methods_name]
            acc_list_start_avg = inner_dict.get('acc_list_start_avg')
            acc_list_start_std = inner_dict.get('acc_list_start_std')
            plot_acc(acc_list_start_avg[c],acc_list_start_std[c],colors[i],markers[i],linestyles[i],methods_name,std = 2)
            max_element = max(max_element,np.max(np.maximum.reduce(acc_list_start_avg[c])))
            min_element = min(min_element,np.min(np.minimum.reduce(acc_list_start_avg[c])))
        plt.xlabel('Iteration', fontsize = 18)
        plt.ylabel('Accuracy', fontsize = 18)
        plt.tick_params(labelsize=18)
        plt.ylim(0.3, 1)
        # plt.ylim(min(0.5,min_element), max_element)
        plt.legend(loc='lower right')
        # plt.title('Accuracy after data distribution shift, d={}'.format(d_list[c]), fontsize = 18)
        file_name = f'acc_d = {d_list[c]}_start_no_std.pdf'
        plt.tight_layout()
        plt.savefig(folder_path+file_name, transparent=True, backend='pdf')

    for c in range(num_d):
        fig = plt.figure(figsize=(15,6))
        for i, methods_name in enumerate(methods):
            inner_dict = data_dict[methods_name]
            model_gaps_avg = inner_dict.get('model_gaps_avg')
            model_gaps_std = inner_dict.get('model_gaps_std')
            plot_model_gap(model_gaps_avg[c],model_gaps_std[c],colors[i],markers[i],linestyles[i],methods_name)
        plt.xlabel('Iteration', fontsize = 18)
        plt.ylabel('Model Consistency', fontsize = 18)
        plt.tick_params(labelsize=18)
        plt.legend(loc='lower right')
        plt.ylim(-1,1)
        # plt.yscale('log')
        # plt.title('Model Consistency, d={}'.format(d_list[c]), fontsize = 18)
        file_name = f'Model_gap_d = {d_list[c]}.pdf'
        plt.tight_layout()
        plt.savefig(folder_path+file_name, transparent=True, backend='pdf')

    for c in range(num_d):
        fig = plt.figure(figsize=(15,6))
        for i, methods_name in enumerate(methods):
            inner_dict = data_dict[methods_name]
            model_gaps_avg = inner_dict.get('model_gaps_avg')
            model_gaps_std = inner_dict.get('model_gaps_std')
            plot_model_gap(model_gaps_avg[c],model_gaps_std[c],colors[i],markers[i],linestyles[i],methods_name,std = 2)
        plt.xlabel('Iteration', fontsize = 18)
        plt.ylabel('Model Consistency', fontsize = 18)
        plt.tick_params(labelsize=18)
        plt.legend(loc='lower right')
        plt.ylim(-1,1)
        # plt.yscale('log')
        # plt.title('Model Consistency, d={}'.format(d_list[c]), fontsize = 18)
        file_name = f'Model_gap_d = {d_list[c]}_no_std.pdf'
        plt.tight_layout()
        plt.savefig(folder_path+file_name, transparent=True, backend='pdf')

    plt.close('all')
    current_time = datetime.now()
    current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print("Plot Completion Time:", current_time_str)

    df = pd.DataFrame()
    descriptions = []
    for i in range(num_d):
        descriptions.append(f'd = {d_list[i]}')
    for i, methods_name in enumerate(methods):
        inner_dict = data_dict[methods_name]
        model_gaps_avg = inner_dict.get('model_gaps_avg')
        means = np.mean(model_gaps_avg[:,20:], axis=1)
        stds = np.std(model_gaps_avg[:,20:], axis=1)
        data = [f"{np.round(mean, decimals=3)} $\\pm$ {np.round(std, decimals=3)}" for mean, std in zip(means, stds)]
        df[methods_name] = data
    df.insert(0, ' ', descriptions)
    # df.rename(index={0: 'Mean and Standard diviation of Model Gap after 20 steps'}, inplace=True)
    df_1 = df.T
    df_1.columns = df_1.iloc[0]  # 将第一行作为列名
    df_1 = df_1.drop(df_1.index[0])  # 删除原先的第一行
    df_1.to_csv(folder_path+'gap_mean_std_after_20.csv')

    df = pd.DataFrame()
    descriptions = []
    for i in range(num_d):
        descriptions.append(f'd = {d_list[i]}')
    for i, methods_name in enumerate(methods):
        inner_dict = data_dict[methods_name]
        acc_list_start_avg = inner_dict.get('acc_list_start_avg')
        means = np.mean(acc_list_start_avg[:,20:], axis=1)
        stds = np.std(acc_list_start_avg[:,20:], axis=1)
        data = [f"{np.round(mean, decimals=3)} $\\pm$ {np.round(std, decimals=3)}" for mean, std in zip(means, stds)]
        df[methods_name] = data
    df.insert(0, ' ', descriptions)
    # df.rename(index={0: 'Mean and Standard diviation of Accuracy after 20 steps'}, inplace=True)
    df_1 = df.T
    df_1.columns = df_1.iloc[0]  # 将第一行作为列名
    df_1 = df_1.drop(df_1.index[0])  # 删除原先的第一行
    df_1.to_csv(folder_path+'acc_mean_std_after_20.csv') #, index=False

    current_time = datetime.now()
    current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print("Table Completion Time:", current_time_str)

    # df = pd.DataFrame()
    # descriptions = []
    # for i in range(num_d):
    #     descriptions.append(f'd = {d_list[i]}')
    # for file_key, inner_dict in data_dict.items():
    #     acc_list_start_avg = inner_dict.get('acc_list_start_avg')
    #     means = np.mean(acc_list_start_avg, axis=1)
    #     stds = np.std(acc_list_start_avg, axis=1)
    #     data = [f"{np.round(mean, decimals=4)} $\\pm$ {np.round(std, decimals=4)}" for mean, std in zip(means, stds)]
    #     df[file_key] = data

    # df.insert(0, 'Description', descriptions)
    # df.rename(index={0: 'Mean and Standard diviation of Accuracy'}, inplace=True)
    # df.to_csv('acc_mean_std.csv', index=False)
        

if __name__ == "__main__":
    plot_fig(num_iters,d_list)
