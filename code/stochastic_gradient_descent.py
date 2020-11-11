
import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
from linear_regression import Regression
from sklearn.linear_model import SGDRegressor

class StochasticDescent:
    def __init__(self,

        X, y, cost_func, theta0=None ,batch_size=32, n_epochs=500,
        t0=0.01, t1=1, v0=1, gamma=0, lam=0, schedule='constant'):
        """
        Class for finding optimal set of weights for a regression or classification model
        using stochastic mini-batch gradient descent.

        Args:
            X (matrix): input design matrix of shape (n_datapoints, n_features).
            y (array/matrix): Target of shape (n_datapoints) for regression.
                Must be in one-hot form for classification.
            cost_func (string): Desired cost function.
                Options: 'MSE', 'cross-entropy'
            theta0 (array/matrix): Initial weights. Default set to NONE
                If not specified, will initialize randomly.
                Required shape for regression: (n_features)
                Required shape for classification: (n_features, n_classes)
            batch_size (int): Size of mini-batches. Default set to 32
            n_epochs (int): Number of epochs to iterate through. Default set to 500.
            t0 (float): Top argument for learning rate. Default set to 0.01
                If learning rate is constant, t0 will represent the learning rate.
            t1 (float): Bottom argument for learning rate. Default set to 1.
                Will only be used in the adaptive learning schedule
            v0 (float): Starting point for momentum. Default set to 1
            gamma (float): Momemntum parameter. Takes values between 0 and 1. Default set to 0.
            lam (float): L2 norm, or penalty. Default set to 0.
            schedule (string): Desired learning schedule. Default set to 'constant'
                options: 'constant', 'adaptive'

        Attributes:
            theta  (array/matrix): Complete set of weights.
            errors (array): MSE as function of epochs.
                For checking convergence.

        """

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
        self.v0         = v0                # initial momentum
        self.lam        = lam               # L2-norm

        # Fetch gradient function and learning schedule
        self.fetch_funcs(cost_func, schedule)

        # Set seed
        np.random.seed(42)

    def fit_regression(self, generate_theta=True):
        """
        Optimize weights using SGD for regression

        Args:
            generate_teta (boolean): Default set to True.
                Will initialize weights randomly

        Returns:
            None.
        """

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

        for epoch in range(1,self.n_epochs+1):
            for i in range(self.n_batch):

                # Fetch random batch
                batch_index = np.random.randint(self.n_batch)
                Xi = X_batches[batch_index]
                yi = y_batches[batch_index]

                # Calculate model and gradients
                z = Xi @ self.theta
                gradients = self.gradient(z, Xi, yi)

                # Calculate learning rate
                self.t = epoch*self.n_batch + i
                learning_rate = self.learning_schedule()

                # Calulate momentum and update beta
                v = self.gamma*v + learning_rate * gradients
                self.theta = self.theta - v

            #self.theta_array[epoch-1] = self.theta
            self.errors[epoch-1] = np.mean( (self.y-self.X@self.theta)**2 )
        #plt.plot(epochs, self.errors)
        #plt.show()


    def fit_classifier(self, generate_theta=True):
        """
        Optimize weights for classification using SGD

        Args:
            generate_teta (boolean): Default set to True.
                Will initialize weights randomly

        Returns:
            None.
        """
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

                # Calculate model and gradients
                self.z = self.softmax( Xi @ self.theta)
                gradients = self.gradient(self.z, Xi, yi)

                # Calculate learning rate
                self.t = epoch*self.n_batch + i
                learning_rate = self.learning_schedule()

                # Calulate momentum and update beta
                v = self.gamma*v + learning_rate * gradients
                self.theta = self.theta - v

    def create_mini_batches(self):
        """
        Shuffles and splits data into minibatches

        Args:
            None

        Returns:
            X_new (matrix): Shuffled and split design matrix
            y_new (array/matrix): Shuffled and split target
        """
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
        """
        Fetch cost function and learning schedule for the class

        Args:
            cost_func (string): Desired cost function.
                Options: 'MSE', 'cross-entropy'
            schedule (string): Desired learning schedule.
                options: 'constant', 'adaptive'

        Returns:
            None
        """
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
        """ Adaptive learning schedule """
        return self.t0/ (self.t + self.t1)

    def constant_schedule(self):
        """ Constant learning schedule """
        return self.t0

    def mse_gradient(self, z, X, y):
        """ Gradient of MSE with l2 norm """
        return 2/self.n_batch * (X.T @ (z - y)) + (2*self.lam * self.theta)

    def cross_entropy_gradient(self, z, X, y):
        """ Gradient of cross-entropy with l2-norm """
        return -(X.T @ (y - z) - self.lam * self.theta) / self.n_batch

    def softmax(self, t):
        """ Softmax """
        exponent = np.exp(t)
        return exponent/np.sum(exponent, axis=1, keepdims=True)
