import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
from linear_regression import Regression
from sklearn.linear_model import SGDClassifier
from stochastic_gradient_descent import stochastic_descent

class LogisticRegressionOwn:
    def __init__(self):
        self.beta       = None  # Complete set of coefficients with intercept
        self.intercept_ = None  # Array of intercept
        self.coef_      = None  # Coefficients without intercept

    def softmax(self, z):
        return 1 / (1 - np.exp(-z))

    def cross_entropy(self, X, beta, y, lam):
        term1 = y * (X @ beta)
        term2 = np.log(1 + np.exp( X @ beta ))
        return - np.sum( term1 - term2 )

    def fit(self, X, y, lam):
        t0 = 1e-3
        t1 = 10
        gamma = 0
        batch_size = 5
        epochs = 100

        # Using our own SGD method for fit
        SGD = stochastic_descent()
        SGD.fit(X, y, batch_size, t0, t1,
        gamma, epochs, lam, self.cross_entropy)

        # Using sklearn's SGD method for comparison
        sgdkit = SGDClassifier(loss='perceptron')
        sgdkit.fit(X, y)

        self.beta       = SGD.beta
        self.coef_      = sgdkit.coef_
        self.intercept_ = sgdkit.intercept_

    def predict(self, X):
        print(np.mean(X @ self.beta))
        return self.sigmoid(X @ self.beta)
