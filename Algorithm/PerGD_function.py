import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import sem
from collections import deque
import time
import csv
import sys

# matplotlib.rcParams['pdf.fonttype'] = 42

def clip_coord(z):
    R = 1
    if z > R:
        return R
    elif z < -R:
        return -R
    else:
        return z

def clip(theta):
    return np.array([clip_coord(z) for z in theta])

def mean0(theta):
    mu0 = 1
    return mu0

def mean1(theta):
    mu1 = -1
    eps = 3
    return mu1 - eps * theta[1]

def shift_dist(n, theta):
    s0 = 0.5
    s1 = 0.5
    g = 0.5
    X = np.ones((n, 2))
    Y = np.ones(n)
    for i in range(n):
        if np.random.rand() <= g:
            X[i, 1] = s1 * np.random.randn() + mean1(theta)
            Y[i] = 1
        else:
            X[i, 1] = s0 * np.random.randn() + mean0(theta)
            Y[i] = 0
    return X, Y

def h(x, theta):
    """
    Logistic model output on x with params theta.
    x should have a bias term appended in the 0-th coordinate.
    1 / (1 + exp{-x^T theta})
    """
    z = np.dot(x, theta)
    if z < -700:
        z = -700
    return 1. / (1. + np.exp(-z))

def loss(x, y, theta):
    reg = 1e-2
    eps = 1e-8
    """
    Cross entropy loss on (x, y) with params theta.
    x should have a bias term appended in the 0-th coordinate.
    """
    return -y * np.log(h(x, theta)+eps) - (1 - y) * np.log(1 - h(x, theta)+eps) + (reg / 2) * np.linalg.norm(theta) ** 2

def est_performative_loss(N, theta):
    X, Y = shift_dist(N, theta)
    return np.mean([loss(x, y, theta) for x, y in zip(X, Y)])

def ce_grad(x, y, theta):
    reg = 1e-2
    return (h(x, theta) - y) * x + reg * theta

def grad1(X, Y, theta):
    n = len(Y)
    d = X.shape[1]
    grad = np.zeros(d)
    for x, y in zip(X, Y):
        grad += ce_grad(x, y, theta)
    return grad / n

def hessian(X, theta, reg):
    """
    Computes the Hessian of the loss on X at theta with ridge regularization reg.
    """
    n = len(X)
    d = len(X[0, :])
    h_vec = np.array([h(x, theta) for x in X])
    w = h_vec * (1 - h_vec)
    
    hess = np.zeros((d, d))
    for i in range(n):
        hess += np.outer(w[i] * X[i], X[i])
    hess += n * reg * np.eye(d)
    return hess

def gradient(X, Y, theta):
    """
    Computes the gradient of the loss on X, Y at theta with ridge regularization reg.
    """
    reg = 1e-2
    n = len(Y)
    h_vec = np.array([h(x, theta) for x in X])
    grad = X.T @ (h_vec - Y) + n * reg * theta
    return grad

def approx_f(X, Y):
    print(np.mean(X[Y == 1][:, 1]))
    print(np.mean(X))
    return np.mean(X[Y == 1][:, 1]) # np.mean(X)

def approx_grad_f(means, thetas):
    dmeans = np.array([m - means[-1] for m in means])
    dthetas = np.array([t - thetas[-1] for t in thetas])
    
    return np.linalg.pinv(dthetas) @ dmeans

def grad2(X, Y, means, thetas, s1):
    """
    X, Y should be the data resulting from thetas[-1]
    """
    n      = len(Y)
    theta  = thetas[-1]
    grad_f = approx_grad_f(means, thetas)
    
    pos_X = X[Y == 1]
    loss_vec  = np.array([loss(x, 1, theta) for x in pos_X])
    
    x_minus_f = np.array([x - means[-1] for x in pos_X[:, 1]])
    
    return (np.dot(loss_vec, x_minus_f) / (n * s1 ** 2)) * grad_f, grad_f

