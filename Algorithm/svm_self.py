import numpy as np
import cvxopt 
from cvxopt import matrix, solvers
import kernel
import time
from numpy import nonzero,dot
import random

class svm_no_b:
    def __init__(self,C = 1,kernelType = 'linear',gamma = 1, R = 1000):
        self.C = C
        self.kernelType = kernelType
        self.gamma = gamma
        self.support_alpha = None
        self.support_indices = None
        self.support_vector = None
        self.support_label = None
        self.R = R
        

    def calculate_max_norm(self,data):
        max_norm = 0
        norms = np.linalg.norm(data, axis=1)
        sorted_indices = np.argsort(norms)[::-1]
        sorted_norms = norms[sorted_indices]
        Q1 = np.percentile(sorted_norms, 25)
        Q3 = np.percentile(sorted_norms, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        cleaned_data = sorted_norms[(sorted_norms >= lower_bound) & (sorted_norms <= upper_bound)]
        # for d in data:
        #     norm = np.linalg.norm(d)
        #     if norm > max_norm:
        #         max_norm = norm
        max_norm = cleaned_data[0]
        return max_norm*max_norm

    def train(self,X,y):
        cvxopt.solvers.options['show_progress'] = False

        # change y from other value to -1
        y[y != 1] = -1

        n_sample = np.shape(X)[0]

        # self.x = np.array(X)
        # self.label = np.array(y).transpose()
        # self.m = np.shape(X)[0]
        # self.n = np.shape(X)[1]
        # self.alpha = np.array(np.zeros(self.m),dtype='float64')
        # self.eCache = np.array(np.zeros((self.m,2)))
        self.R = self.calculate_max_norm(X)

        start_time = time.time()  
        self.K = kernel.compute_kernel_matrix(X,X,self.kernelType,self.gamma) + self.R
        time_taken = time.time() - start_time  
        # print("Compute Kernel took: ", round(time_taken, 6), "seconds")

        P = matrix(np.outer(y, y) * self.K)
        q = matrix(-np.ones(n_sample))
        G = np.concatenate((np.eye(n_sample),-1*np.eye(n_sample)),axis=0)
        G = matrix(G)
        h = matrix(np.concatenate(((self.C/n_sample) * np.ones(n_sample), np.zeros(n_sample)),axis=0)) #/n_samples
        A = None
        b = None

        start_time = time.time()  
        sol = solvers.qp(P, q, G, h, A, b)
        alphas = np.array(sol['x'])
        # SVIndex,SV,SVAlpha,SVLabel = self.smoP()
        time_taken = time.time() - start_time  
        # print("Solve QP took:       ", round(time_taken, 6), "seconds")
        
        support_condition = (alphas > 0.00001) & (alphas < (self.C/n_sample)) #/n_samples

        self.support_alpha = alphas[support_condition]
        self.support_indices = np.where(support_condition)[0]
        self.support_vector = X[self.support_indices]
        self.support_label = y[self.support_indices]

        # if self.kernelType == 'linear':
        temp = self.support_alpha * self.support_label
        self.w = np.dot(temp.T,self.support_vector)

    def norm(self):
        temp  = kernel.compute_kernel_matrix(self.support_vector,self.support_vector,kernel=self.kernelType,parameter=self.gamma) + self.R
        temp2 = self.support_alpha * self.support_label
        result = np.dot(np.dot(temp2.T,temp),temp2)
        return np.sqrt(result)
                
    def predict(self, X):
        temp = kernel.compute_kernel_matrix(self.support_vector,X,kernel=self.kernelType,parameter=self.gamma) + self.R
        temp2 = self.support_alpha * self.support_label
        score = np.dot(temp2.T,temp)
        pred_label = np.sign(score)
        return score,pred_label

