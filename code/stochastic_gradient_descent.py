
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
        self.n_epochs   = int(n_epochs)     # number of epochs
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
        epochs = np.linspace(1,self.n_epochs, self.n_epochs)
        self.errors = np.zeros(len(epochs))
        self.theta_array = np.zeros(len(epochs))
        print(self.n_epochs)
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

            self.theta_array[epoch-1] = self.theta
            self.errors[epoch-1] = np.mean( (self.y-self.X@self.theta)**2 )
        plt.plot(epochs, self.errors)
        plt.show()


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
        return 2/self.n_batch * X.T @ (z - y) + (2*self.lam * self.theta)

    def cross_entropy_gradient(self, z, X, y):
        return -(X.T @ (y - z) - self.lam * self.theta) / self.n_batch

    def softmax(self, t):
        exponent = np.exp(t)
        return exponent/np.sum(exponent, axis=1, keepdims=True)

"""
import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
from linear_regression import Regression
from sklearn.linear_model import SGDRegressor

np.random.seed(45)

class StochasticDescent:
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
        errors = np.zeros(len(epochs))
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
            errors[epoch-1] = np.mean((y - X @ beta)**2)
        plt.plot(epochs,errors)
        plt.show()


        # Store final set of coefficients and intercept
        self.beta = beta
        self.intercept_ = beta[0]
        self.coef_ = beta[1:]
"""
