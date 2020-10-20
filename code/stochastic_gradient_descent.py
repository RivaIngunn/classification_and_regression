import numpy as np
import matplotlib.pyplot as plt
from linear_regression import Regression
np.random.seed(42)

class stochastic_descent:
    def __init__(self):
        self.beta = None        # Complete set of coefficients with intercept
        self.intercept_ = None  # Array of intercept
        self.coef_ = None       # Coefficients without intercept

    def step_length(self, t, t0, t1):
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

    def fit(self, X, y, batch_size):
        n, feats = X.shape
        batch_size = 5
        n_batch = int(n/batch_size)
        n_epochs = 500
        t0 = 1
        t1 = 10

        beta = np.random.randn(feats)

        learning_rate = t0/t1


        for epoch in range(1,n_epochs+1):
            # Perform Gradient Descent on minibatches
            for i in range(n_batch):
                # Break data into mini batches
                X_batches, y_batches = self.create_mini_batches(X, y, n_batch)
                batch_index = np.random.randint(n_batch)
                # Fetch batches and calculate gradient
                Xi = X_batches[batch_index]
                yi = y_batches[batch_index]
                gradients = 2.0/n_batch * Xi.T @ (Xi @ beta - yi) # Derivated 1/n_batch * (Xi @ beta - yi)^2 with respect to beta
                
                print ('k', batch_index)
                # Calculate step length
                t = epoch*n_batch + i
                learning_rate = self.step_length(t, t0, t1)
                beta = beta - learning_rate*gradients
                # print(beta, "Epoch = ", epoch)


        #print(beta)
        # Store final set of coefficients and intercept
        self.beta = beta
        self.intercept_ = beta[0]
        self.coef_ = beta[1:]

if __name__ == "__main__":
    reg = Regression()
    reg.dataset_franke(400)
    maxdeg = 10
    degrees = np.linspace(1,maxdeg,maxdeg, dtype = 'int')

    train_error = np.zeros(maxdeg)
    test_error  = np.zeros(maxdeg)

    batch_size = 5
    for i, deg in enumerate(degrees):
        reg.design_matrix(deg)
        reg.split(reg.X, reg.f)

        SDG = stochastic_descent()
        SDG.fit(reg.X_train, reg.f_train, batch_size)

        z_tilde = reg.X_train @ SDG.beta
        z_pred  = reg.X_test  @ SDG.beta

        train_error[i] = np.mean( (z_tilde - reg.f_train)**2 )
        test_error[i]  = np.mean( (z_pred  - reg.f_test )**2 )

plt.plot(degrees, train_error, label = 'train_error')
plt.plot(degrees, test_error, label = 'test_error')
plt.xlabel('deg')
plt.ylabel('Error')
plt.legend()
plt.show()
