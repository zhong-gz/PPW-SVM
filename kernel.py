import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

def compute_kernel_matrix(X1,X2,kernel = 'linear', parameter = 1):
    if kernel == 'rbf':
        return rbf_kernel(X1,X2, parameter)
    elif kernel == 'linear': 
        return linear_kernel(X1,X2)
    else:
        return linear_kernel(X1,X2)

def linear_kernel(X1,X2):
    kernel_matrix = np.dot(X1, X2.T)
    return kernel_matrix

def rbf_kernel(X1,X2, gamma):

    norm_X1 = np.sum(X1 ** 2, axis=1, keepdims=True)

    norm_X2 = np.sum(X2 ** 2, axis=1, keepdims=True)
    dist = norm_X1 + norm_X2.T - 2 * np.dot(X1, X2.T)
    K = np.exp(-dist / 2*(gamma**2))
    return K