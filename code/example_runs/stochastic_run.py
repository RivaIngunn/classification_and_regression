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

def MSE(model, target):
    return np.mean( (model - target)**2 )

def cost_ridge(X, beta, y, lam):
    y_tilde = X @ beta

    return (np.sum( (y - y_tilde)**2 ) + np.sum(lam*beta**2))/2.

def evaluate_blearning_rate():
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

            # Fit using own SGD method-
            SGD = StochasticDescent(reg.X_train, reg.f_train,
            cost_func='MSE', batch_size = batch_size,t0=1, schedule='adaptive')
            SGD.fit_regression()

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
"""
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
"""
def evaluate_learning_rate_old():
        # Generate Franke data
        reg = Regression()
        reg.dataset_franke(1000)
        deg = 8

        # Generate design matrix and split data
        reg.design_matrix(deg)
        reg.split(reg.X, reg.f, scale=False)


        # Set up parameters for SGD
        t0_array = np.linspace(5,15,10)
        batch_size=2**6

        t1 = 50
        lam = 1e-3
        epochs = 500
        gamma = 0

        # Set up array for plotting
        R2_scores = np.zeros( len(t0_array) )
        error_kit = np.zeros( len(t0_array) )

        for i, t0_val in enumerate(t0_array):

            # Fit using own SGD method-
            SGD = StochasticDescent()
            SGD.fit(reg.X_train, reg.f_train, batch_size, t0_val, t1,
            gamma=0, n_epochs=300, lam=0, loss_func=MSE_OLS)

            # Fit using sklearn's SGD method
            sgdreg = SGDRegressor(max_iter = epochs, penalty=None, eta0=t0_val/t1)
            sgdreg.fit(reg.X_train, reg.f_train)

            # Calculate models
            z_tilde = reg.X_train @ SGD.beta
            z_kit = reg.X_train @ sgdreg.coef_

            # Calculate errors
            print(R2_score(reg.f_train, z_tilde))
            #print(np.mean(SGD.beta))
            R2_scores[i] = R2_score(reg.f_train, z_tilde)
            error_kit[i] = R2_score(reg.f_train, z_kit)
        plt.plot(t0_array, R2_scores)
        plt.plot(t0_array, error_kit)
        plt.show()
def evaluate_learning_rate_new():
        # Generate Franke data
        reg = Regression()
        reg.dataset_franke(1000)
        deg = 8

        # Generate design matrix and split data
        reg.design_matrix(deg)
        reg.split(reg.X, reg.f, scale=False)


        # Set up parameters for SGD
        t0_array = np.linspace(1,10,10)
        batch_size=32

        t1 = 50
        lam = 1e-3
        epochs = 500
        gamma = 0

        # Set up array for plotting
        R2_scores = np.zeros( len(t0_array) )
        error_kit = np.zeros( len(t0_array) )

        for i, t0_val in enumerate(t0_array):

            # Fit using own SGD method-
            SGD = StochasticDescent(reg.X_train, reg.f_train, cost_func='MSE',
            batch_size=batch_size, n_epochs=epochs, t0=t0_val, t1=t1, schedule='adaptive')
            SGD.fit_regression()

            # Fit using sklearn's SGD method
            sgdreg = SGDRegressor(max_iter = epochs, penalty=None, eta0=t0_val/t1)
            sgdreg.fit(reg.X_train, reg.f_train)

            # Calculate models
            z_tilde = reg.X_test @ SGD.theta
            z_kit = reg.X_test @ sgdreg.coef_

            # Calculate errors
            print(R2_score(reg.f_test, z_tilde))
            #print(np.mean(SGD.beta))
            R2_scores[i] = R2_score(reg.f_test, z_tilde)
            error_kit[i] = R2_score(reg.f_test, z_kit)
        plt.plot(t0_array, R2_scores)
        plt.plot(t0_array, error_kit)
        plt.show()
def evaluate_batches_new():
        # Generate Franke data
        reg = Regression()
        reg.dataset_franke(1000)
        deg = 8

        # Generate design matrix and split data
        reg.design_matrix(deg)
        reg.split(reg.X, reg.f, scale=False)

        # Set up parameters for SGD
        t0 = 10
        batch_sizes = np.arange(6,40,4)
        print(len(batch_sizes))

        t1 = 50
        lam = 1e-3
        epochs = 1000
        gamma = 0

        # Set up array for plotting
        R2_scores = np.zeros( len(batch_sizes) )
        error_kit = np.zeros( len(batch_sizes) )

        for i, batch_size in enumerate(batch_sizes):

            # Fit using own SGD method-
            SGD = StochasticDescent(reg.X_train, reg.f_train, cost_func='MSE',
            batch_size=batch_size, n_epochs=epochs, t0=t0, t1=t1, schedule='adaptive')
            SGD.fit_regression()

            # Fit using sklearn's SGD method
            sgdreg = SGDRegressor(max_iter = epochs, penalty=None, eta0=t0/t1)
            sgdreg.fit(reg.X_train, reg.f_train)

            # Calculate models
            z_tilde = reg.X_test @ SGD.theta
            z_kit = reg.X_test @ sgdreg.coef_

            # Calculate errors
            print(R2_score(reg.f_test, z_tilde))
            #print(np.mean(SGD.beta))
            R2_scores[i] = R2_score(reg.f_test, z_tilde)
            error_kit[i] = R2_score(reg.f_test, z_kit)
        plt.plot(batch_sizes, R2_scores)
        plt.plot(batch_sizes, error_kit)
        plt.show()
def evaluate_momentum_new():
        # Generate Franke data
        reg = Regression()
        reg.dataset_franke(1000)
        deg = 8

        # Generate design matrix and split data
        reg.design_matrix(deg)
        reg.split(reg.X, reg.f, scale=False)

        # Set up parameters for SGD
        t0 = 10
        batch_size = 24
        gamma_array=np.linspace(0,0.7,8)

        t1 = 50
        lam = 1e-3
        epochs = 20
        gamma = 0

        # Set up array for plotting
        R2_scores = np.zeros( len(gamma_array) )
        error_kit = np.zeros( len(gamma_array) )

        for i, gamma in enumerate(gamma_array):

            # Fit using own SGD method-
            SGD = StochasticDescent(reg.X_train, reg.f_train, cost_func='MSE',
            batch_size=batch_size, n_epochs=epochs, t0=t0, t1=t1, gamma=gamma, schedule='adaptive')
            SGD.fit_regression()

            # Fit using sklearn's SGD method
            sgdreg = SGDRegressor(max_iter = epochs, penalty=None, eta0=t0/t1)
            sgdreg.fit(reg.X_train, reg.f_train)

            # Calculate models
            z_tilde = reg.X_test @ SGD.theta
            z_kit = reg.X_test @ sgdreg.coef_

            # Calculate errors
            print(R2_score(reg.f_test, z_tilde))
            #print(np.mean(SGD.beta))
            R2_scores[i] = R2_score(reg.f_test, z_tilde)
            error_kit[i] = R2_score(reg.f_test, z_kit)

            epoch_array = np.linspace(1, epochs, epochs)
            plt.plot(epoch_array, SGD.errors, label=r'$\gamma = $%.1f'%gamma)

        plt.legend()
        plt.show()

        plt.plot(gamma_array, R2_scores)
        plt.plot(gamma_array, error_kit)
        plt.show()
def comparison():
    reg = Regression()
    reg.dataset_franke(1000)
    maxdeg=10

    # Setting up arrays for plotting
    train_error_OLS = np.zeros(maxdeg)
    test_error_OLS  = np.zeros(maxdeg)

    train_error_SGD = np.zeros(maxdeg)
    test_error_SGD  = np.zeros(maxdeg)

    degrees = np.linspace(1, maxdeg, maxdeg, dtype=int)
    for i, deg in enumerate(degrees):
        # Load OLS models
        reg.design_matrix(deg)
        reg.split(reg.X, reg.f)
        f_tilde_OLS, f_pred_OLS = reg.OLS(reg.X_train, reg.X_test, reg.f_train)

        # Load SGD models
        SGD = StochasticDescent(reg.X_train, reg.f_train, cost_func='MSE',
        batch_size=2, n_epochs=1000, t0=1, t1=10, gamma=0.5, lam=0.1, schedule='adaptive')
        SGD.fit_regression()
        f_tilde_SGD = reg.X_train @ SGD.theta
        f_pred_SGD  = reg.X_test  @ SGD.theta

        # Store MSE for OLS
        train_error_OLS[i] = MSE(reg.f_train, f_tilde_OLS)
        test_error_OLS[i]  = MSE(reg.f_test, f_pred_OLS)

        # Store MSE for SGD
        train_error_SGD[i] = MSE(reg.f_train, f_tilde_SGD)
        test_error_SGD[i]  = MSE(reg.f_test, f_pred_SGD)

    # Plotting errors
    plt.plot(degrees, train_error_OLS, label='OLS train')
    plt.plot(degrees, test_error_OLS,  label='OLS test')
    plt.plot(degrees, train_error_SGD, label='SGD train')
    plt.plot(degrees, test_error_SGD,  label='SGD test')
    plt.legend()
    plt.show()

#evaluate_learning_rate_new()
#evaluate_batches_new()
#evaluate_mini_batches()
#evaluate_momentum_new()
comparison()
