import numpy as np

class stochastic_descent:
    def __init__(self):
        self.epoch = epoch

    def step_length(self, t,t0,t1):
        return t0/(t+t1)

    def descent(self):
        n = 100
        M = 5
        m = int(n/M)
        n_epochs = 500
        t0 = 1.0
        t1 = 10

        gamma_j = t0/t1
        j = 0
        for epoch in range(1,n_epochs+1):
            for i in range(m):
                k = np.random.randint(m) # Pick random minibatch
                # compute gradient
                # Compute new suggestion for beta
                t = epoch*m + i
                gamma_j = step_length(t_t0,t1)
                j += 1
