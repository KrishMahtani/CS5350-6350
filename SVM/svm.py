from math import exp
import numpy as np
import scipy.optimize

class PrimalSVM:
    def __init__(self, X, y, lr_schedule, C, epochs=10):
        self.weights = np.ndarray
        self.train(X, y, lr_schedule, C, epochs)

    def train(self, X, y, lr_schedule, C, epochs=10):
        X = np.insert(X, 0, [1]*len(X), axis=1)
        self.weights = np.zeros_like(X[0])

        for e in range(epochs):
            lr = lr_schedule(e)
            idxs = np.arange(len(X))
            np.random.shuffle(idxs)

            for i in idxs:
                if y[i]*np.dot(self.weights, X[i]) <= 1:
                    zeroed_bias = self.weights.copy()
                    zeroed_bias[0] = 0
                    self.weights = self.weights - lr*zeroed_bias + lr*C*y[i]*X[i]
                else:
                    self.weights = (1-lr)*self.weights

    def predict(self, X) -> np.ndarray:
        X = np.insert(X, 0, [1]*len(X), axis=1)
        pred = lambda d : np.sign(np.dot(self.weights, d))
        return np.array([pred(xi) for xi in X])

def _gaussian(x, y, gamma):
    return exp(-(np.linalg.norm(x-y, ord=2)**2) / gamma)


class DualSVM:
    def __init__(self, X, y, C, kernel = "dot", gamma=None):
        self.wstar = np.ndarray
        self.bstar = float
        self.support = []
        self.train(X, y, C, kernel, gamma)

    def train(self, X, y, C, kernel = "dot", gamma=None):

        def inner(a, X, y):
            ymat = y * np.ones((len(y), len(y)))
            amat = a * np.ones((len(a), len(a)))

            if kernel == 'dot':
                xvals = (X@X.T)
            if kernel == 'gaussian':
                xvals = X**2 @ np.ones_like(X.T) - 2*X@X.T + np.ones_like(X) @ X.T**2 
                xvals = np.exp(-( xvals / gamma))

            vals = (ymat*ymat.T) * (amat*amat.T) * xvals
            return 0.5*np.sum(vals) - np.sum(a)

        constraints = [
            {
                'type': 'ineq',
                'fun': lambda a : a # a > 0 constraint
            },
            {
                'type': 'ineq',
                'fun': lambda a: C - a # a < c constraint
            },
            {
                'type': 'eq',
                'fun': lambda a: np.dot(a, y) # sum_i a_i*y_y = 0 constraint
            }
        ]
        
        # minimize inner function to find Lagrange multipliers a*
        res = scipy.optimize.minimize(inner, x0=np.zeros(shape=(len(X),)), args=(X, y), method='SLSQP', constraints=constraints, tol=0.01)

        # use these values to calculate weights
        self.wstar = np.zeros_like(X[0])
        for i in range(len(X)):
            self.wstar += res['x'][i]*y[i]*X[i]

        # and bias
        self.bstar = 0
        if kernel == 'dot':
            for j in range(len(X)):
                self.bstar += y[j] - np.dot(self.wstar, X[j])
        if kernel == 'gaussian':
            for j in range(len(X)):
                self.bstar += y[j] - _gaussian(self.wstar, X[j], gamma)
        self.bstar /= len(X)

        THRESH = 1e-10
        for i, a in enumerate(res['x']):
            if a > THRESH:
                self.support.append(X[i])

    def predict(self, X, kernel = "dot", gamma=None) -> np.ndarray:
        if kernel == 'dot':
            pred = lambda d : np.sign(np.dot(self.wstar, d) + self.bstar)
        if kernel == 'gaussian':
            pred = lambda d : np.sign(_gaussian(self.wstar, d, gamma) + self.bstar)
        return np.array([pred(xi) for xi in X])
