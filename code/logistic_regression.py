import autograd.numpy as np
from stochastic_gradient_descent import stochastic_descent
from sklearn import datasets
from sklearn.model_selection import train_test_split

class MultinomialRegression:
    """ Multinomial regression for classification """
    def __init__(self, X, y, eta, batch_size, epochs, iterations, lam):
        self.X = X # input
        self.y = self.to_categorical_numpy(y) # target in one-hot form

        # Define all sizes
        self.datapoints  = X.shape[0]      # number of datapoints
        self.feats       = X.shape[1]      # number of features
        self.n_classes   = self.y.shape[1] # number of classes

        # Define parameters for gradient descent
        self.eta        = eta        # learning rate
        self.batch_size = batch_size # size of minibatches
        self.epochs     = epochs     # number of epochs
        self.iterations = iterations # number of iterations per epoch
        self.lam        = lam        # l2 norm

        # Initiate weights
        self.weights = np.random.randn(self.feats, self.n_classes)

    def calculate_step(self, X):
        z = X @ self.weights
        output = self.softmax(z)
        gradient = self.cross_entropy_gradient(X_batch, y_batch, output)
        return gradient

    def fit(self):
        """ Use SGD to optimize weights """
        data_indices = np.arange(self.datapoints)

        # Perform stochastic gradient descent
        for i in range(self.epochs):
            for j in range(self.iterations):

                # Select random area of input
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )
                # Create minibatches
                X_batch = self.X[chosen_datapoints]
                y_batch = self.y[chosen_datapoints]

                # Calculate gradient
                z = X_batch @ self.weights
                output = self.softmax(z)
                gradient = self.cross_entropy_gradient(X_batch, y_batch, output)

                # Update weights
                self.weights -= self.eta * gradient

    def predict(self, X):
        z = X @ self.weights
        output = self.softmax(z)
        return np.argmax(output, axis=1)

    def accuracy_score(self, model, target):
        I = np.where(model == target)[0]
        return len(I) / len(target)

    def softmax(self, t):
        exponent = np.exp(t)
        return exponent/np.sum(exponent, axis=1, keepdims=True)

    def cross_entropy_gradient(self, X, y, O):
        return -(X.T @ (y - O) - self.lam*self.weights)/ self.datapoints

    def to_categorical_numpy(self, integer_vector):
        n_inputs = len(integer_vector)
        n_categories = np.max(integer_vector) + 1
        onehot_vector = np.zeros((n_inputs, n_categories))
        onehot_vector[range(n_inputs), integer_vector] = 1

        return onehot_vector
