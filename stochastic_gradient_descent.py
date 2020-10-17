import numpy as np
from linear_regression import Regression

class stochastic_descent:
    def __init__(self):
        self.epoch = 2

    def step_length(self, t, t0, t1):
        return t0/(t+t1)

    def fit(self, X, y):
        n = X.shape[0]
        feats  = X.shape[1]
        batch_size = 5
        n_batch = int(n/batch_size)
        print(n_batch)
        n_epochs = 500
        t0 = 1.0
        t1 = 10

        beta = np.random.randn(feats)

        learning_rate = t0/t1
        j = 0

        for epoch in range(1,n_epochs+1):
            # Perform Gradient Descent on minibatches
            for i in range(n_batch):
                # Select random index representing batch
                # and then select corresponding regions of X and y
                random_index = np.random.randint(n_batch)
                Xi = X[random_index:random_index+1]
                yi = y[random_index:random_index+1]

                gradients = 2.0/n_batch * Xi.T @ (Xi @ beta - yi) # Derivated 1/n_batch * (Xi @ beta - yi)^2 with respect to beta

                t = epoch*n_batch + i
                learning_rate = self.step_length(t, t0, t1)
                beta = beta - learning_rate*gradients
                j += 1
                print(gradients)
        self.beta = beta # Store final set of coefficients and intercept
        self.intercept_ = beta[0] # Store intercept
        self.coef_ = beta[1:] # Store coefficients


X = np.array([
[1,2,3,4,5,6],
[4,5,6,7,8,9]
])
y = np.array([3,4])

reg = Regression()
reg.dataset_franke(400)
reg.design_matrix(8)

SDG = stochastic_descent()
SDG.fit(reg.X,reg.f)
