import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from datetime import datetime

# Import classes
import sys
sys.path.insert(0,"..")
from linear_regression import Regression
from stochastic_gradient_descent import StochasticDescent
from logistic_regression import MultinomialRegression
from load_mnist import LoadMNIST

# Load MNIST dataset
dat = LoadMNIST()

# Logistic regression using sklearn  -----------------------
logreg = LogisticRegression()
logreg.fit(dat.x_train, dat.y_train)
pred = logreg.predict(dat.x_test)

# Checking accuracy
score = logreg.score(dat.x_test, dat.y_test)
print(score)

# Using our own logistic regression ------------------------

# Define parameters for SGD
batch_size = 2**6
epochs = 100
eta = 0.01
lam = 0.1

# Use our Logistic method to make prediction
mulreg = MultinomialRegression(dat.x_train, dat.y_train, batch_size=batch_size, epochs=epochs, t0=eta, lam=lam)
mulreg.fit()
pred2 = mulreg.predict(dat.x_test)

# Checking accuracy
score2 = mulreg.accuracy_score(pred2, dat.y_test)

print(dat.y_test[:30])
print(pred2[:30])
print(score2)

# Random sampling
def random_search(iterations, print_vals=False):
    """
    Performs random search for a given amount of iterations
    and writes out the best result to logistic_mark.txt.
    """
    # Set up parameter lists
    epochs = 500
    batch_sizes = [4, 16, 32, 64, 128]
    learning_rates = np.logspace(-2, 1, 10)
    gamma_array = np.linspace(0, 0.99, 10)
    penalties = np.logspace(-5, -1, 10)

    # Storage of parameters used and accuracy score
    eta_arr         = np.zeros(iterations)
    batch_size_arr  = np.zeros(iterations)
    gamma_arr       = np.zeros(iterations)
    lam_arr         = np.zeros(iterations)
    scores          = np.zeros(iterations)

    current_iteration = 1
    for i in range(iterations):
        # Fetch random values
        eta_arr[i]         = np.random.choice(learning_rates)
        batch_size_arr[i]  = np.random.choice(batch_sizes)
        gamma_arr[i]       = np.random.choice(gamma_array)
        lam_arr[i]         = np.random.choice(penalties)

        # Print for debugging
        if print_vals:
            print("eta: ", eta_arr[i])
            print("B: ", batch_size_arr[i])
            print("gamma: ", gamma_arr[i])
            print("lam: ", lam_arr[i])

        # Do logistic regression
        mulreg = MultinomialRegression(dat.x_train, dat.y_train,
        batch_size=batch_size_arr[i], epochs=epochs, t0=eta_arr[i], lam=lam_arr[i], gamma=gamma_arr[i])
        mulreg.fit()
        pred = mulreg.predict(dat.x_test)

        # Calculate accuracy
        scores[i] =  mulreg.accuracy_score(pred, dat.y_test)

        # Print progress
        print("Iteration: %i/%i"%(current_iteration, iterations))
        current_iteration += 1

    # Set nan values to 0
    scores = np.nan_to_num(scores)
    idx = np.argmax(scores)

    # Store best values in benchmark file
    filename = "../../benchmarks/logistic_mark.txt"
    outfile = open(filename, 'a')
    outfile.write("---- Random search: %i iterations ----" %iterations)
    outfile.write("\nAccuracy score: %f"    %scores[idx])
    outfile.write("\nLearning rate: %.3e"   %eta_arr[idx])
    outfile.write("\nBatch size: %i"        %batch_size_arr[idx])
    outfile.write("\nGamma: %.3f"           %gamma_arr[idx])
    outfile.write("\nPenalty: %.3e"         %lam_arr[idx])
    outfile.write("\n\n")

    print("Accuracy score: ", scores[idx])
    print("Learning rate: ", eta_arr[idx])
    print("Batch size: ", batch_size_arr[idx])
    print("Gamma: ", gamma_arr[idx])
    print("Penalty: ", lam_arr[idx])

random_search(100)
