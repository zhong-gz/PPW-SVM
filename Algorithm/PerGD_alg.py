import numpy as np
from scipy.stats import sem
from collections import deque
from PerGD_function import shift_dist,approx_f,grad1,clip,grad2,est_performative_loss
from sklearn.linear_model import LogisticRegression

class PerGD:
    def __init__(self,H = 50, lr = 0.1):
        self.s1 = 0.5
        self.H  = H
        self.lr = lr # learning rate
        # self.history = deque()
        # self.history.append(self.theta.copy())
        self.grad_fs = deque()
        # self.g2s     = deque()
        self.thetas = deque(maxlen = H + 1)
        self.means  = deque(maxlen = H + 1)

    def train(self,X,y_ture):
        d = X.shape[1]
        self.theta = 2 * np.random.rand(d) - 1
        if len(self.thetas) == 0:
            self.thetas.append(self.theta.copy())
        Y = np.copy(y_ture)
        Y[Y != 1] = 0
        if len(self.thetas) < 2:
            model = LogisticRegression()
            model.fit(X, Y)
            self.theta = model.coef_.T
            # self.thetas.append(self.theta.copy())
        else:
            if len(self.thetas) < 2:
                self.means.append(approx_f(X, Y))
                grad = grad1(X, Y, self.theta)

                self.theta = clip(self.theta - self.lr * grad).copy()
                # self.history.append(self.theta.copy())
                self.thetas.append(self.theta.copy())
            else:
                self.means.append(approx_f(X, Y))
                g2, grad_f = grad2(X, Y, self.means, self.thetas, self.s1)
                grad = grad1(X, Y, self.theta) + g2
                self.grad_fs.append(grad_f)
                # self.g2s.append(g2)

                self.theta = clip(self.theta - self.lr * grad).copy()
                # self.history.append(self.theta.copy())
                self.thetas.append(self.theta.copy())
            
    def predict(self,X):
        score = np.dot(X, self.theta)
        h = self.sigmoid(score)
        predictions = (h >= 0.5).astype(int)
        predictions[predictions != 1] = -1
        return score,predictions
    
    def sigmoid(self,z):
        z[z < -700] = -700
        return 1 / (1 + np.exp(-z))