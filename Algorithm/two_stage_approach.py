import numpy as np
from numpy.linalg import lstsq

class two_stage_algo:
    def __init__(self,X_base,y_base,tol = 1e-6):
        self.X_base = np.copy(X_base)
        self.y_base = np.copy(y_base)
        self.y_base[self.y_base != 1] = 0
        self.theta = None
        self.tol = tol
        self.mu = np.zeros_like(X_base[0,:])
        self.X_shift = None # np.zeros_like(X_base)
        self.theta_list = np.zeros_like(X_base)

    def calculate_performative_effect(self):
        d = self.X_base.shape[1]
        for i in range(d):
            x = self.X_shift[:, i]
            y = self.theta_list[:, i]
            if not np.all(x == 0):
                self.mu[i] = np.sum(x * y) / np.sum(x**2)
        self.mu = np.nan_to_num(self.mu, nan=0)

    def train(self,X,y_ture):
        y = np.copy(y_ture)
        y[y != 1] = 0
        n = len(y)
        d = self.X_base.shape[1]

        if self.X_shift is None:
            self.X_shift = X - self.X_base
        else:
            self.X_shift = np.concatenate((self.X_shift, X - self.X_base), axis=0)

        self.calculate_performative_effect()

        X_b = np.c_[np.ones((len(self.X_base), 1)),self.X_base]
        theta_init = np.random.randn(d+1)
        learning_rate = 0.01
        num_iterations = 10000
        theta_final, cost_history = self.gradient_descent(X_b, y, self.mu, theta_init, learning_rate, num_iterations)
        # if np.isnan(theta_final).any():
        #     theta_final = self.theta_list[-1] + np.random.normal(loc=0, scale=1, size=len(self.theta_list[-1]))
        self.theta = np.copy(theta_final)
        self.theta_list = np.concatenate((self.theta_list, np.tile(theta_final,(n, 1))), axis=0)
        return theta_final

    def predict(self,X):
        score = np.dot(X, self.theta)
        if np.isnan(score).any():
            score = np.random.normal(loc=0, scale=0.1, size=len(score))
        h = self.sigmoid(score)
        predictions = (h >= 0.5).astype(int)
        predictions[predictions != 1] = -1
        return score,predictions

    def sigmoid(self,z):
        z[z < -700] = -700
        return 1 / (1 + np.exp(-z))

    def compute_cost(self,X, y, theta):
        m = len(y)
        h = self.sigmoid(np.dot(X, theta))
        epsilon = 1e-8
        cost = (-1/m) * np.sum(y*np.log(h+ epsilon) + (1-y)*np.log(1-h+ epsilon))
        return cost

    def gradient_descent(self,X, y,mu , theta, learning_rate, num_iterations):
        m = len(y)
        cost_history = []
        mu = np.insert(mu, 0, 0)

        for i in range(num_iterations):
            h = self.sigmoid(np.dot(X + mu*theta, theta))
            gradient = np.dot((X+2*mu*theta).T, (h - y)) / m
            theta -= learning_rate * gradient
            cost = self.compute_cost(X+ mu*theta, y, theta)
            cost_history.append(cost)
            if (i > 2) and (np.abs(cost_history[-1] - cost_history[-2]) < 1e-3/m):
                return theta[1:], cost_history

        return theta[1:], cost_history
