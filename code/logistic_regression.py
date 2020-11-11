import autograd.numpy as np
from stochastic_gradient_descent import StochasticDescent
from sklearn import datasets
from sklearn.model_selection import train_test_split

class MultinomialRegression:
    """ Multinomial regression for classification """
    def __init__(self, X, y, batch_size=32, epochs=500,
        t0=0.01, t1=1, v0=1, gamma=0, lam=0, schedule='constant'):
        """
        Classifying a dataset

        Args:
            X (matrix): Input design matrix of shape (n_datapoints, n_features)
            y (array): Array of shape (n_datapoints) with target labels
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
            weights (matrix): Optimized weights of shape (n_features, n_classes)

        """
        self.X = X # input
        self.y = self.to_categorical_numpy(y) # target in one-hot form

        # Define all sizes
        self.datapoints  = X.shape[0]      # number of datapoints
        self.feats       = X.shape[1]      # number of features
        self.n_classes   = self.y.shape[1] # number of classes

        # Define parameters for gradient descent
        self.batch_size = batch_size        # size of minibatches
        self.epochs     = epochs            # number of epochs
        self.t0         = t0                # top learning rate parameter
        self.t1         = t1                # bottom learning rate parameter
        self.gamma      = gamma             # momentum parameter
        self.v0         = v0                # initial momentum
        self.lam        = lam               # L2-norm

        # Set schedule
        self.schedule = schedule

        # Initiate weights
        self.weights = np.random.randn(self.feats, self.n_classes)

    def fit(self):
        """ Fit using SGD """
        SGD = StochasticDescent(self.X, self.y, cost_func='cross entropy', theta0=self.weights, batch_size=self.batch_size,
        n_epochs=self.epochs, t0=self.t0, t1=self.t1, v0=self.v0, gamma=self.gamma, lam=self.lam, schedule=self.schedule)
        SGD.fit_classifier(generate_theta=False)
        self.weights = SGD.theta

    def predict(self, X):
        """ Apply weights to an input and return predicted labels"""
        z = X @ self.weights
        output = self.softmax(z)
        return np.argmax(output, axis=1)

    def accuracy_score(self, model, target):
        """
        Calculate the accurcy score of a model

        Args:
            model(Array): Predicted labels
            target(Array): Target labels
        Returns:
            Accuracy score (float): Percentage of correct tabels

        """
        I = np.where(model == target)[0]
        return len(I) / len(target)

    def softmax(self, t):
        """ Softmax """
        exponent = np.exp(t)
        return exponent/np.sum(exponent, axis=1, keepdims=True)

    def cross_entropy_gradient(self, X, y, O):
        """ Derivative of cross-entropy with respect to the weights """
        return -(X.T @ (y - O) - self.lam*self.weights)/ self.datapoints

    def to_categorical_numpy(self, integer_vector):
        """
        Converts target to one-hot form.
        This method was made my Morten H. Jensen
        """
        n_inputs = len(integer_vector)
        n_categories = np.max(integer_vector) + 1
        onehot_vector = np.zeros((n_inputs, n_categories))
        onehot_vector[range(n_inputs), integer_vector] = 1

        return onehot_vector
