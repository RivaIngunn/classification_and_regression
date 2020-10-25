import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor

import sys
sys.path.insert(0,"..")
from linear_regression import Regression
from stochastic_gradient_descent import stochastic_descent

def MSE(X, beta, y):
    return np.mean(( X @ beta - y)**2)

def evaluate_learning_rate():
    reg = Regression()
    reg.dataset_franke(200)

    batch_size = 5
    t0 = np.logspace(-3,0,10)
    gamma = 0

    maxdeg = 2
    degrees = np.linspace(1,maxdeg,maxdeg, dtype = 'int')

    for deg in degrees:
        error     = np.zeros(t0.shape)
        error_kit = np.zeros(t0.shape)

        for i, t in enumerate(t0):
            reg.design_matrix(deg)
            reg.split(reg.X, reg.f)

            SDG = stochastic_descent()
            SDG.fit(reg.X_train, reg.f_train, batch_size, t, 10, gamma, epochs = 300, loss_func = MSE)

            sgdreg = SGDRegressor(max_iter = 100, penalty=None, eta0=t/10)
            sgdreg.fit(reg.X_train, reg.f_train, )

            z_tilde = reg.X_train @ SDG.beta
            z_kit = reg.X_train @ sgdreg.coef_

            error[i] = np.mean((z_tilde - reg.f_train)**2)
            error_kit[i] = np.mean((z_kit - reg.f_train)**2)

        plt.plot(np.log10(t0), error, label = "Deg: %i"%deg)
        plt.plot(np.log10(t0), error_kit, label = "Kit Deg: %i"%deg)
    plt.legend()
    plt.show()

evaluate_learning_rate()
