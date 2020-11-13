import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt

import sys
sys.path.insert(0,"..")
from logistic_regression import MultinomialRegression

""" Keep in mind that there is a softmax test in test_sgd.py"""
def test_onehot():
    """ Checking if the one hot representation is correct """
    for i in range(1):
        # Generate random labels and input
        n = 500
        y = np.random.randint(0,10, n)
        x = np.random.randn(n,10)

        # Convert to one hot
        logreg = MultinomialRegression(x, y, 0.01, 32, 500, 10, 0)
        onehot = logreg.to_categorical_numpy(y)

        # Convert back from onehot and calculate difference
        y2 = np.argmax(onehot, axis=1)
        diff = y - y2

        assert np.sum(diff) == 0, "onehot representation incorrect"
