import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor

import sys
sys.path.insert(0,"..")
from linear_regression import Regression
from stochastic_gradient_descent import StochasticDescent
from plotter import plotter

def R2_score(y, y_tilde):
    top = np.mean( (y - y_tilde)**2 )
    bottom = np.var(y)
    return 1- top/bottom

def MSE(model, target):
    return np.mean( (model - target)**2 )

def cost_ridge(X, beta, y, lam):
    y_tilde = X @ beta

    return (np.sum( (y - y_tilde)**2 ) + np.sum(lam*beta**2))/2.

def load_franke(n, deg):
        # Generate Franke data
        global reg
        reg = Regression()
        reg.dataset_franke(n)

        # Generate design matrix and split data
        reg.design_matrix(deg)
        reg.split(reg.X, reg.f)

        return reg.X_train, reg.f_train, reg.X_test, reg.f_test

def evaluate_learning_rate_OLS(t0_array, batch_array, filename, t1=1, epochs=500, gamma=0, lam=0, schedule='constant'):
    """ Evaluate constant learning rate and batch sizes """
    # Load franke data
    x_train, y_train, x_test, y_test = load_franke(10000, 6)
    # Produce model and calculate R2 score
    f_tilde_OLS, f_pred_OLS = reg.OLS(x_train, x_test, y_train)
    R2_OLS = R2_score(y_test, f_pred_OLS)

    # Set up arrays for plotting
    R2_scores  = np.zeros( len(t0_array) )
    error_kit  = np.zeros( len(t0_array) )
    R2_OLS_arr = np.ones(  len(t0_array) ) * R2_OLS
    plt.style.use('seaborn-whitegrid')

    # Keeping track of iterations
    total_iterations = len(t0_array) * len(batch_array)
    current_iteration = 1

    # Iterate through values
    for batch_size in batch_array:
        for i, t0_val in enumerate(t0_array):
            # Fit using own SGD method
            SGD = StochasticDescent(x_train, y_train, cost_func='MSE',
            batch_size=batch_size, n_epochs=epochs, t0=t0_val, t1=t1, schedule=schedule)
            SGD.fit_regression()

            # Fit using sklearn's SGD method
            sgdreg = SGDRegressor(max_iter = epochs, penalty=None, eta0=t0_val)
            sgdreg.fit(x_train, y_train)

            # Calculate models
            z_tilde = x_test @ SGD.theta
            z_kit = x_test @ sgdreg.coef_

            # Calculate errors
            R2_own = R2_score(y_test, z_tilde)
            R2_kit = R2_score(y_test, z_kit)

            #print(np.mean(SGD.beta))
            R2_scores[i] = R2_own
            error_kit[i] = R2_kit

            # Print progress
            print("Iteration: %i/%i"%(current_iteration, total_iterations))
            print("Batch: %i, LR: %.2e, R2: %.4f"%(batch_size, t0_val, R2_own))
            current_iteration += 1

        # Remove divergent and extreme underfitting regions
        indices = np.where(R2_scores >= 0)
        kit_idx = np.where(error_kit >= 0)
        plt.plot(np.log10(t0_array[indices]), R2_scores[indices], label='Batch size: %i'%batch_size)

    # Plot R2-scores for OLS and sklearn's SGD for comparison
    plt.plot(np.log10(t0_array[kit_idx]), error_kit[kit_idx], "--", label='sklearn')
    plt.plot(np.log10(t0_array), R2_OLS_arr, ":", color="r", label='OLS')

    # Generate plot
    title = "R2-scores as function of learning rate"
    xlabel = r"learning rate: $log_{10}(\eta)$"
    ylabel = r"$R^2$"

    plot = plotter()
    plot.single_plot_show(xlabel, ylabel, title, save=True, filename=filename)

def evaluate_learning_rate_ridge(t0_array, lam_array, filename, batch_size=32, epochs=500, t1=10, schedule='constant'):
    # Load franke data
    x_train, y_train, x_test, y_test = load_franke(10000, 6)

    # Produce model and calculate R2 score
    f_tilde_OLS, f_pred_OLS = reg.ridge(x_train, x_test, y_train, lam=1e-3)
    R2_OLS = R2_score(y_test, f_pred_OLS)

    # Set up parameter arrays for SGD
    t0_array  = np.logspace(-3, -1, 10)

    # Set up remaining parameters
    t1 = 1
    epochs = 500
    gamma = 0

    # Set up arrays for plotting
    R2_scores  = np.zeros( len(t0_array) )
    error_kit  = np.zeros( len(t0_array) )
    R2_OLS_arr = np.ones(  len(t0_array) ) * R2_OLS

    # Keeping track of iterations
    total_iterations = len(t0_array) * len(lam_array)
    current_iteration = 1

    # Iterate through values
    for lam in lam_array:
        for i, t0_val in enumerate(t0_array):
            # Fit using own SGD method
            SGD = StochasticDescent(x_train, y_train, cost_func='MSE',
            batch_size=batch_size, n_epochs=epochs, t0=t0_val, t1=t1, lam=lam, schedule='constant')
            SGD.fit_regression()

            # Fit using sklearn's SGD method
            sgdreg = SGDRegressor(max_iter = epochs, penalty=None, eta0=t0_val/t1)
            sgdreg.fit(x_train, y_train)

            # Calculate models
            z_tilde = x_test @ SGD.theta
            z_kit = x_test @ sgdreg.coef_

            # Calculate errors
            R2_own = R2_score(y_test, z_tilde)
            R2_kit = R2_score(y_test, z_kit)

            #print(np.mean(SGD.beta))
            R2_scores[i] = R2_own
            error_kit[i] = R2_kit

            # Print progress
            print("Iteration: %i/%i"%(current_iteration, total_iterations))
            current_iteration += 1

        # Remove divergent and extreme underfitting regions
        indices = np.where(R2_scores >= 0)
        plt.plot(np.log10(t0_array[indices]), R2_scores[indices], label=r'$\lambda$: %.2e'%lam)

    #plt.plot(np.log10(t0_array), error_kit)
    plt.plot(np.log10(t0_array), R2_OLS_arr, label='OLS')

    # Generate plot
    title = "R2-scores as function of learning rate and penalty"
    xlabel = r"learning rate: $log_{10}(\eta)$"
    ylabel = r"$R^2$"

    plot = plotter()
    plot.single_plot_show(xlabel, ylabel, title, save=True, filename=filename)

def evaluate_momentum():
        # Load franke data
        x_train, y_train, x_test, y_test = load_franke(10000, 6)

        # Set up parameters for SGD
        t0 = 0.05
        batch_size = 32
        gamma_array=np.linspace(0, 0.5, 5)
        t1 = 1
        epochs = 2000

        # Set up array for plotting
        R2_scores = np.zeros( len(gamma_array) )
        error_kit = np.zeros( len(gamma_array) )
        plt.style.use('seaborn-whitegrid')

        # Keep track of iterations
        total_iterations = len(gamma_array)
        current_iteration = 1

        for i, gamma in enumerate(gamma_array):

            # Fit using own SGD method-
            SGD = StochasticDescent(x_train, y_train, cost_func='MSE',
            batch_size=batch_size, n_epochs=epochs, t0=t0, t1=t1, gamma=gamma, schedule='constant')
            SGD.fit_regression()

            # Fit using sklearn's SGD method
            sgdreg = SGDRegressor(max_iter = epochs, penalty=None, eta0=t0/t1)
            sgdreg.fit(x_train, y_train)

            # Calculate models
            z_tilde = x_test @ SGD.theta
            z_kit = x_test @ sgdreg.coef_

            # Calculate errors
            R2_scores[i] = R2_score(y_test, z_tilde)
            error_kit[i] = R2_score(y_test, z_kit)

            epoch_array = np.linspace(1, epochs, epochs)
            plt.plot(epoch_array, SGD.errors, label=r'$\gamma = $%.1f'%gamma)

            # Print progress
            print("Iteration: %i/%i"%(current_iteration, total_iterations))
            current_iteration += 1

        # Generate plot
        title = "MSE convergence in SGD with momentum"
        xlabel = "epoch"
        ylabel = "MSE"
        filename = "momentum_convergence"

        plot = plotter()
        plt.ylim(0.14,0.25)
        plot.single_plot_show(xlabel, ylabel, title, save=True, filename=filename)
def do_random_search():
    """ Perform random search to find the optimal parameters """
    # Load franke data
    x_train, y_train, x_test, y_test = load_franke(10000, 6)

    # Set up ranges of hyperparameters
    t0_array = np.linspace(1e-3, 1, 100)
    batch_array = [16, 32, 64, 128, 256]
    gamma_array = np.linspace(0, 0.9, 100)
    penalty_array = np.logspace(-3, -1, 100)
    epochs = 500

    # Set up array for loss
    iterations = 10
    R2_array = np.zeros(iterations)

    # Keeping track of iterations
    current = 1

    # Perform random search
    for i in range(iterations):
        t0          = np.random.choice(t0_array, 1)
        batch_size  = np.random.choice(batch_array, 1)
        gamma       = np.random.choice(gamma_array, 1)
        lam         = np.random.choice(penalty_array, 1)

        # Perform SGD
        SGD = StochasticDescent(x_train, y_train, cost_func='MSE',
        batch_size=batch_size, n_epochs=epochs, t0=t0, gamma=gamma, lam=lam, schedule='constant')
        SGD.fit_regression()

        # Calculate loss
        model = x_test @ SGD.theta
        R2_array[i] = R2_score(y_test, model)

        # Update iteration
        print("Iteration %i/%i"%(current, iterations))
        current += 1

    print(R2_array)
    print(np.max(R2_array))


# Evaluate constant learning rate and batch size for OLS
t0_array = np.logspace(-5, 1, 15)
batch_array = [4, 8, 32, 64, 128]
filename = "learning_rate_batch_SGD"
#evaluate_learning_rate_OLS(t0_array, batch_array, filename)

# Evaluate adaptive learning rate for OLS
#t0_array = np.linspace(100, 1000, 5)
#batch_array = [8]#16, 32, 64, 128]
#filename = "adaptive_learning_rate_SGD"
#evaluate_learning_rate_OLS(t0_array, batch_array, filename, epochs=500, t1=10, schedule='adaptive')

#Evaluate constant learning rate for Ridge
t0_array = np.logspace(-5, 1, 10)
lam_array = np.logspace(-8, -3, 5)
filename = "learning_rate_ridge"
#evaluate_learning_rate_ridge(t0_array, lam_array, filename)

# Evaluate momentum for OLS
evaluate_momentum()

# Put it together

# Do random search
#do_random_search()
