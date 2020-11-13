import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt

import sys
sys.path.insert(0,"..")
from linear_regression import Regression
from stochastic_gradient_descent import StochasticDescent

def test_convergence():
    """
    Checking that the model converges for such a low learning rate that
    it gets trapped in a local minimum. Test essensially confirms that
    gradient descent is performed by calculating the variance of the last
    50 epochs and comparing it with the variance of the entire epoch space.
    """
    # Load, scale and split franke data
    reg = Regression()
    reg.dataset_franke(10000)
    reg.design_matrix(6)
    reg.split(reg.X, reg.f)

    # Perform SGD and fetch errors
    SGD = StochasticDescent(reg.X_train, reg.f_train, t0=0.01, cost_func='MSE')
    SGD.fit_regression()
    err = SGD.errors
    epoch_array = np.linspace(1,500,500)

    # Calculate variance for last 50 epochs
    error_variance = np.var(err[450:])
    total_variance = np.var(err)
    relative_variance = error_variance / total_variance

    # Assert
    assert relative_variance < 0.01, "No convergence was achieved"

def test_softmax_probabilites():
    """ Checks if the sum of probabilities from Softmax is equal to 1"""
    for i in range(10):
        print(1)
        # Set up data
        n = 100
        X = np.random.randn(n, n)
        y = np.random.randint(0, 10, n)

        # Convert to one-hot
        onehot = np.zeros((n, 10))
        onehot[range(n), y] = 1
        y = onehot

        # Stochastic Gradient Descent
        SGD = StochasticDescent(X, y, cost_func='cross entropy')
        SGD.fit_classifier()

        # Calculate sum and difference
        prob_sum =  np.sum(SGD.z, axis=1)
        diff = prob_sum - 1

        # Assert
        assert (diff < 1e-10).all(), "Sum of probabilities does not equal 1"
