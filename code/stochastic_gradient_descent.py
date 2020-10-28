import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
from linear_regression import Regression
from sklearn.linear_model import SGDRegressor

np.random.seed(45)

class stochastic_descent:
    def __init__(self):
        self.beta       = None  # Complete set of coefficients with intercept
        self.intercept_ = None  # Array of intercept
        self.coef_      = None  # Coefficients without intercept

    def learning_schedule(self, t, t0, t1):
        return t0/(t+t1)

    def create_mini_batches(self, X, y, n_batches):
        # Shuffle data
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)

        X_new = X[indices]
        y_new = y[indices]

        # Split into minibatches
        X_new = np.array_split(X_new, n_batches)
        y_new = np.array_split(y_new, n_batches)

        return X_new, y_new

    def fit(self, X, y, batch_size, t0, t1, gamma, n_epochs, lam, loss_func, beta0=None):
        # Fetch relevant parameters for minibatches
        n, feats = X.shape
        n_batch = int(n/batch_size)

        # Setting up a starting point
        if beta0 == None:
            beta = np.random.randn(feats)
        else:
            beta = beta0
        learning_rate = t0/t1
        v = 1

        # Break data into mini batches
        X_batches, y_batches = self.create_mini_batches(X, y, n_batch)
        intercepts = np.zeros(n_epochs)
        epochs = np.linspace(1, n_epochs, n_epochs)
        for epoch in range(1,n_epochs+1):
            # Perform Gradient Descent on minibatches
            for i in range(n_batch):
                # Fetch random batch
                batch_index = np.random.randint(n_batch)
                Xi = X_batches[batch_index]
                yi = y_batches[batch_index]

                # Calculate gradients analytically and using autograd
                #gradients = 2/n_batch * Xi.T @ ((Xi @ beta) - yi)
                gradient_func = grad(loss_func, 1)
                gradients = gradient_func(Xi, beta, yi, lam)

                # Calculate learning rate
                t = epoch*n_batch + i
                learning_rate = self.learning_schedule(t, t0, t1)

                # Calulate momentum and update beta
                v = gamma*v + learning_rate * gradients
                beta = beta - v


        # Store final set of coefficients and intercept
        self.beta = beta
        self.intercept_ = beta[0]
        self.coef_ = beta[1:]

"""autopep"""
