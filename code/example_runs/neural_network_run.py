import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0,"..")
from linear_regression import Regression
from neural_network import FFNN
from load_mnist import LoadMNIST

"""X = np.array([
[1,2,3,4],
[1,3,3,5],
[5,3,2,2],
[7,8,9,1],
[3,4,6,2]
])

y = np.array([
[1,2,3,4,5],
[3,4,5,6,2]
])
y = y.T #reshape(-1,1

neunet = FFNN(X, y, 6, 3, hidden_function = 'sigmoid', output_function = 'relu')
neunet.train()
a = neunet.feed_forward(X)
neunet.back_propagation()

delta = np.array([1,2])
z = np.array([4,5,7])
W = np.array([[1,2,3], [1,2,3]])"""
"""
reg = Regression()
reg.dataset_franke(1000)
reg.design_matrix(5)
reg.split(reg.X, reg.f)
y = reg.f_train.reshape(-1,1)
neunet = FFNN(reg.X_train, y, 2, 5, cost_func='SSR', hidden_func='sigmoid', output_func='linear')
neunet.train()

model = neunet.predict(reg.X_test)
plt.plot(model)
plt.plot(reg.f_test)
plt.show()
"""
dat = LoadMNIST()
def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1

    return onehot_vector
y = to_categorical_numpy(dat.y_train)
x = dat.x_train
neunet = FFNN(x, y, 2, 10, cost_func='cross-entropy', hidden_func='sigmoid', output_func='softmax',
batch_size=32, eta=0.01, lam=0.0)
#neunet.feed_forward(x)
#neunet.back_propagation()
neunet.train()
model = neunet.predict(dat.x_train, classify=True)
print(model[:20])
print(dat.y_train[:20])
I = np.where(model == dat.y_train)[0]
print(len(I) / len(dat.y_train))
