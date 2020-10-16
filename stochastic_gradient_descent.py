import numpy as np

class stochastic_descent:
    def __init__(self):
        self.epoch = 2

    def step_length(self, t, t0, t1):
        return t0/(t+t1)

    def descent(self, X, y):
        n = X.shape[0]
        feats  = X.shape[1]
        batch_size = 5
        n_batch = int(n/batch_size)
        n_epochs = 500
        t0 = 1.0
        t1 = 10

        beta = np.random.randn(feats)

        learning_rate = t0/t1
        j = 0

        for epoch in range(1,n_epochs+1):
            for i in range(n_batch):
                random_index = np.random.randint(n_batch)
                xi = X[random_index:random_index+1]
                yi = y[random_index:random_index+1]

                loss = (xi*beta - yi)**2
                gradient = np.gradient(xi)

                t = epoch*n_batch + i
                learning_rate = self.step_length(t, t0, t1)
                beta = beta - learning_rate*gradient
                j += 1

X = np.array([
[1,2,3,4,5,6],
[4,5,6,7,8,9]
])
y = np.array([3,4])

SDG = stochastic_descent()
SDG.descent(X,y)
