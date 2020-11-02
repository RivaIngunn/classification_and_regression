import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor

import sys
sys.path.insert(0,"..")
from linear_regression import Regression
from stochastic_gradient_descent import StochasticDescent

def R2_score(y, y_tilde):
    top = np.mean( (y - y_tilde)**2 )
    bottom = np.var(y)
    return 1- top/bottom

def MSE_OLS(X, beta, y, lam):
    return np.mean( (X @ beta - y)**2 )

def cost_ridge(X, beta, y, lam):
    y_tilde = X @ beta

    return (np.sum( (y - y_tilde)**2 ) + np.sum(lam*beta**2))/2.

def evaluate_learning_rate():
    """ Loop through minibatch sizes and initial learning rates """
    # Generate Franke data
    reg = Regression()
    reg.dataset_franke(1000)
    deg = 8

    # Set up parameters for SGD
    t0_array = np.linspace(5,15,10)
    batch_size_array = np.arange(2,24,2)

    t1 = 50
    lam = 1e-3
    epochs = 500
    gamma = 0

    # Set up array for plotting
    R2_scores = np.zeros( (len(batch_size_array), len(t0_array)) )

    # Loop through minebatch sizes
    for i, batch_size in enumerate(batch_size_array):
        # Loop trough initial learning rates
        for j, t0_val in enumerate(t0_array):

            # Generate design matrix and split data
            reg.design_matrix(deg)
            reg.split(reg.X, reg.f, scale=False)

            # Fit using own SGD method
            SGD = stochastic_descent()
            SGD.fit(reg.X_train, reg.f_train, batch_size, t0_val, t1,
            gamma, epochs, lam, loss_func = cost_ridge)

            # Fit using sklearn's SGD method
            sgdreg = SGDRegressor(max_iter = epochs, penalty=None, eta0=t0_val/t1)
            #sgdreg.fit(reg.X_train, reg.f_train)

            # Calculate models
            z_tilde = reg.X_train @ SGD.beta
            #z_kit = X @ sgdreg.coef_

            # Calculate errors
            print(R2_score(reg.f_train, z_tilde))
            #print(np.mean(SGD.beta))
            R2_scores[i,j] = R2_score(reg.f_train, z_tilde)
            #error_kit[i] = R2_score(reg.f_test, z_kit)

    reg.heatmap(R2_scores, "R2_scores")

def evaluate_mini_batches():
    # Generate Franke data
    reg = Regression()
    reg.dataset_franke(200)

    # Set up parameters for SGD
    deg = 2
    batch_size_array = np.linspace(1,100,10)
    t0 = 1.0
    gamma = 0
    epochs = 10
    t1 = 10
    lam = 1e-3

    error     = np.zeros(batch_size_array.shape)
    error_kit = np.zeros(batch_size_array.shape)

    # Loop trough initial learning rates
    for i, batch_size in enumerate(batch_size_array):

        # Generate design matrix and split data
        reg.design_matrix(deg)
        reg.split(reg.X, reg.f)

        # Fit using own SGD method
        SGD = stochastic_descent()
        SGD.fit(reg.X_train, reg.f_train, batch_size, t0, t1,
        gamma, epochs, lam, loss_func = MSE_OLS)

        # Fit using sklearn's SGD method
        sgdreg = SGDRegressor(max_iter = epochs, penalty=None, eta0=t0/t1)
        sgdreg.fit(reg.X_train, reg.f_train)

        # Calculate models
        z_tilde = reg.X_train @ SGD.beta
        z_kit = reg.X_train @ sgdreg.coef_

        # Calculate errors
        error[i] = np.mean((z_tilde - reg.f_train)**2)
        error_kit[i] = np.mean((z_kit - reg.f_train)**2)

    # Plot
    plt.plot(batch_size_array, error, label = "Deg: %i"%deg)
    plt.plot(batch_size_array, error_kit, label = "Kit Deg: %i"%deg)
    plt.legend()
    plt.show()

reg = Regression()
reg.dataset_franke(1000)
deg = 8
reg.design_matrix(deg)
reg.split(reg.X, reg.f)

SGD = StochasticDescent(reg.X_train, reg.f_train, cost_func='MSE', t0=1, schedule='adaptive')
SGD.fit_regression()
beta = SGD.theta

plt.plot(reg.X_test @ beta)
plt.plot(reg.f_test)
plt.show()
#evaluate_learning_rate()
#evaluate_mini_batches()
