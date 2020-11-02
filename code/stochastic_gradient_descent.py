import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
from linear_regression import Regression
from sklearn.linear_model import SGDRegressor

np.random.seed(45)

class StochasticDescent:
    def __init__(self,
        X, y, cost_func, theta0=None ,batch_size=32, n_epochs=500,
        t0=0.01, t1=1, v0=1, gamma=0, lam=0, schedule='constant'):
        # Store data
        self.X      = X           # design matrix
        self.y      = y           # target
        self.theta0 = theta0      # initial weights
        self.feats  = X.shape[1]  # number of features
        self.n      = X.shape[0]  # number of datapoints

        # Store SGD parameters
        self.batch_size = batch_size        # size of minibatches
        self.n_batch    = int(self.n/self.batch_size) # number of minibatches
        self.n_epochs   = n_epochs          # number of epochs
        self.t          = None              # varying learning rate parameter
        self.t0         = t0                # top learning rate parameter
        self.t1         = t1                # bottom learning rate parameter
        self.gamma      = gamma             # momentum parameter
        self.v0         = v0
        self.lam        = lam               # L2-norm

        # Fetch gradient function and learning schedule
        self.fetch_funcs(cost_func, schedule)

    def fit_regression(self, generate_theta=True):

        # Setting up a starting point
        if generate_theta:
            self.theta = np.random.randn(self.feats)
        else:
            self.theta = self.theta0
        learning_rate = self.t0/self.t1
        v = self.v0

        # Break data into mini batches
        X_batches, y_batches = self.create_mini_batches()

        for epoch in range(1,self.n_epochs+1):
            for i in range(self.n_batch):

                # Fetch random batch
                batch_index = np.random.randint(self.n_batch)
                Xi = X_batches[batch_index]
                yi = y_batches[batch_index]

                # Calculate model
                z = Xi @ self.theta
                # Calculate gradients
                gradients = self.gradient(z, Xi, yi)

                # Calculate learning rate
                self.t = epoch*self.n_batch + i
                learning_rate = self.learning_schedule()

                # Calulate momentum and update beta
                v = self.gamma*v + learning_rate * gradients
                self.theta = self.theta - v

    def fit_classifier(self, generate_theta=True):
        # Setting up a starting point
        if generate_theta:
            self.theta = np.random.randn(self.feats, self.y.shape[1])
        else:
            self.theta = self.theta0

        learning_rate = self.t0 / self.t1
        v = self.v0

        # Break data into mini batches
        X_batches, y_batches = self.create_mini_batches()

        for epoch in range(1,self.n_epochs+1):
            for i in range(self.n_batch):

                # Fetch random batch
                batch_index = np.random.randint(self.n_batch)
                Xi = X_batches[batch_index]
                yi = y_batches[batch_index]

                # Calculate model
                z = self.softmax( Xi @ self.theta)
                # Calculate gradients
                gradients = self.gradient(z, Xi, yi)

                # Calculate learning rate
                self.t = epoch*self.n_batch + i
                learning_rate = self.learning_schedule()

                # Calulate momentum and update beta
                v = self.gamma*v + learning_rate * gradients
                self.theta = self.theta - v

    def create_mini_batches(self):
        """ Split design matrix, target and input into minibatches"""
        # Shuffle data
        indices = np.arange(self.X.shape[0])
        np.random.shuffle(indices)

        X_new = self.X[indices]
        y_new = self.y[indices]

        # Split into minibatches
        X_new = np.array_split(X_new, self.n_batch)
        y_new = np.array_split(y_new, self.n_batch)

        return X_new, y_new

    def fetch_funcs(self, cost_func, schedule):
        """Fetch functions used in descent """
        # Fetching cost function and assigning corresponding gradient
        if cost_func == 'MSE':
            self.gradient = self.mse_gradient
        elif cost_func == 'cross entropy':
            self.gradient = self.cross_entropy_gradient

        # Fetching learning schedule method
        if schedule == 'constant':
            self.learning_schedule = self.constant_schedule
        elif schedule == 'adaptive':
            self.learning_schedule = self.adaptive_schedule

    def adaptive_schedule(self):

        return self.t0/ (self.t + self.t1)

    def constant_schedule(self):
        return self.t0

    def mse_gradient(self, z, X, y):
        return 2/self.n_batch * X.T @ (z - y) + 2*self.lam * self.theta

    def cross_entropy_gradient(self, z, X, y):
        return -(X.T @ (y - z) - self.lam * self.theta) / self.n_batch

    def softmax(self, t):
        exponent = np.exp(t)
        return exponent/np.sum(exponent, axis=1, keepdims=True)
