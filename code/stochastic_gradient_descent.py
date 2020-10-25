import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
from linear_regression import Regression
from sklearn.linear_model import SGDRegressor

np.random.seed(45)

class stochastic_descent:
    def __init__(self):
        self.beta = None        # Complete set of coefficients with intercept
        self.intercept_ = None  # Array of intercept
        self.coef_ = None       # Coefficients without intercept

    def learning_schedule(self, t, t0, t1):
        return t0/(t+t1)

    def create_mini_batches(self, X, y, n_batches):
        # Shuffle data
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)

        X_new = X[indices]
        y_new = y[indices]

        # Split into minibatches
        X_new = np.array_split(X_new, n_batches)
        y_new = np.array_split(y_new, n_batches)

        return X_new, y_new

    def fit(self, X, y, batch_size, t0, t1, gamma, loss_func):
        n, feats = X.shape
        n_batch = int(n/batch_size)
        n_epochs = 100

        # Setting up a starting point
        beta = np.random.randn(feats)
        learning_rate = t0/t1
        v = 1

        errors = np.zeros(n_epochs)
        epochs = np.linspace(1, n_epochs,n_epochs)

        # Break data into mini batches
        X_batches, y_batches = self.create_mini_batches(X, y, n_batch)

        for epoch in range(1,n_epochs+1):
            # Perform Gradient Descent on minibatches
            for i in range(n_batch):
                # Fetch random batch and calculate gradient
                batch_index = np.random.randint(n_batch)
                Xi = X_batches[batch_index]
                yi = y_batches[batch_index]
                gradients = 2/n_batch * Xi.T @ (Xi @ beta - yi) # Derivated 1/n_batch * (Xi @ beta - yi)^2 with respect to beta

                gradient_func = grad(loss_func, 1)
                gradients_ = gradient_func(Xi, beta, yi)

                # Calculate step length
                t = epoch*n_batch + i
                learning_rate = self.learning_schedule(t, t0, t1)

                v = gamma*v + learning_rate*gradient_func(Xi, beta - gamma*v, yi)
                #print(v)

                beta = beta - v#learning_rate*gradients_

            #errors[epoch-1] = np.mean((X @ beta - y)**2)

        #plt.plot(epochs, errors)
        #plt.show()

        #print(beta)
        # Store final set of coefficients and intercept
        self.beta = beta
        self.intercept_ = beta[0]
        self.coef_ = beta[1:]

if __name__ == "__main__":
    def MSE(X, beta, y):
        return np.mean(( X @ beta - y)**2)
    reg = Regression()
    reg.dataset_franke(200)

    batch_size = 32
    t0 = np.logspace(-3,1,10)
    gamma = 0.5

    maxdeg = 2
    degrees = np.linspace(1,maxdeg,maxdeg, dtype = 'int')

    for deg in degrees:
        error     = np.zeros(t0.shape)
        error_kit = np.zeros(t0.shape)

        for i, t in enumerate(t0):
            reg.design_matrix(deg)
            reg.split(reg.X, reg.f)

            SDG = stochastic_descent()
            SDG.fit(reg.X_train, reg.f_train, batch_size, t, 10, gamma, loss_func = MSE)

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
    exit()
