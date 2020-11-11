import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Import classes
import sys
sys.path.insert(0,"..")
from linear_regression import Regression
from stochastic_gradient_descent import StochasticDescent
from logistic_regression import MultinomialRegression
from load_mnist import LoadMNIST

# Load MNIST dataset
dat = LoadMNIST()

# Logistic regression using sklearn
logreg = LogisticRegression()
logreg.fit(dat.x_train, dat.y_train)
pred = logreg.predict(dat.x_test)

# Checking accuracy
score = logreg.score(dat.x_test, dat.y_test)
print(score)

# Using our own logistic regression
batch_size = 2**6
epochs = 100
iterations = 50
eta = 0.01
lam = 0.1

mulreg = MultinomialRegression(dat.x_train, dat.y_train, batch_size=batch_size, epochs=epochs, t0=eta, lam=lam)
mulreg.fit()
pred2 = mulreg.predict(dat.x_test)

X = pred2
y = dat.y_test
score2 = mulreg.accuracy_score(pred2, dat.y_test)

print(dat.y_test[:30])
print(pred2[:30])
print(score2)
