import numpy as np
import matplotlib.pyplot as plt
from linear_regression import Regression
from load_mnist import LoadMNIST

class FFNN:
    def __init__(self, X, y, n_layers, hidden_nodes, cost_func, hidden_func, output_func, batch_size, eta, lam=0):
        # Store input and target
        self.X            = X # input
        self.y            = y # target

        # Define hyperparameters
        self.lam        = lam        # L2 norm
        self.batch_size = batch_size # size of batches
        self.eta        = eta        # learning rate

        # Define all lists
        self.weights = [] # weights
        self.bias    = [] # biases
        self.sizes   = [] # list of nodes per layer

        # Define all sizes
        self.n_layers     = n_layers     # number of layers
        self.hidden_nodes = hidden_nodes # number of nodes per hidden layer
        self.output_nodes = y.shape[1]   # number of nodes in output
        self.feats        = X.shape[1]   # number of features in input
        self.datapoints   = X.shape[0]   # number of datapoints in input

        # Set activation functions and cost
        self.fetch_funcs(cost_func, hidden_func, output_func)

        # Initialize network
        self.generate_architecture()


    def generate_architecture(self):
        """ Set up architecture of network with random weights and biases """
        # Define structure
        self.sizes.append(self.feats)
        for l in range(self.n_layers-1):
            self.sizes.append(self.hidden_nodes)
        self.sizes.append(self.output_nodes)
        print(self.sizes)

        # Loop through layers
        for l in range(1, self.n_layers + 1):
            # Set up random layer weights and small biases
            self.weights.append(np.random.randn( self.sizes[l-1], self.sizes[l] ) *np.sqrt(1/self.sizes[l-1]) )
            self.bias.append(np.full( self.sizes[l], 0.01 ))

    def feed_forward(self, X):
        """ Feed forward neural network and produce output"""
        # Setting up parameters dependent on datapoints
        n_points = X.shape[0]
        self.z     = []
        self.a     = []
        self.delta = []
        for l in range(1, self.n_layers + 1):
            self.z.append(     np.zeros( (self.sizes[l], n_points) ) )
            self.a.append(     np.zeros( (self.sizes[l], n_points) ) )
            self.delta.append( np.zeros( (self.sizes[l], n_points) ) )

        # Fetch weights and biases (for cleaner code)
        weights = self.weights
        bias = self.bias

        # Calculate activations for first layer
        self.z[0] = X @ weights[0] + bias[0]
        self.a[0] = self.hidden_func(self.z[0])

        # Calculate activations for rest of hidden layers
        for l, w in enumerate(weights[1:-1]):
            self.z[l+1] = ( self.z[l] @ w + bias[l+1])
            self.a[l+1] = self.hidden_func(self.z[l+1])

        # Calculate activations for output layer
        self.z[-1] = ( self.z[-2] @ weights[-1] + bias[-1])
        self.a[-1] = self.output_func(self.z[-1])

        # Return output
        return self.a[-1]

    def back_propagation(self):
        """ Perform backpropagation """
        # Calculate output error
        #print(self.y.shape)
        #print(self.a[-1].shape)
        self.delta[-1] = self.output_prime(self.z[-1]) * self.gradient(self.X, self.y, self.a[-1], self.weights[-1], self.lam)
        #print("delta shape: ", self.delta[-1].shape)

        # Calculate errors in hidden layers
        for l in range(2, (self.n_layers + 1)):
            self.delta[-l] = (self.delta[-l + 1] @ self.weights[-l + 1].T) * self.sigmoid_prime(self.z[-l])

        # Gradient descent
        for l in range(1, self.n_layers):
            # Calulate step lengths
            weight_step = self.a[l-1].T @ self.delta[l] / self.datapoints
            bias_step   = np.sum(self.delta[l]) / self.datapoints

            # Add regularization
            weight_step += self.lam * self.weights[l]

            # Update weights and biases
            self.weights[l] = self.weights[l] - self.eta*weight_step
            self.bias[l] = self.bias[l] - self.eta*bias_step

    def predict(self, X, classify=False):
        output = self.feed_forward(X)
        if classify:
            return np.argmax(output, axis=1)
        else:
            return output

    def train(self):
        data_indices = np.arange(self.datapoints)

        for i in range(100):
            for j in range(50):
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )
                X_batch = self.X[chosen_datapoints]
                y_batch = self.y[chosen_datapoints]

                self.feed_forward(self.X)
                self.back_propagation()

    def fetch_funcs(self, cost_func, hidden_func, output_func):
        # Fetching cost function and assigning corresponding gradient
        if cost_func == 'SSR':
            self.gradient = self.cost_derivative
        elif cost_func == 'cross-entropy':
            self.gradient = self.cross_entropy_gradient

        # Fetching activation function for output
        if output_func == 'linear':
            self.output_func  = self.linear
            self.output_prime = self.linear_prime
        elif output_func == 'sigmoid':
            self.output_func  = self.sigmoid
            self.output_prime = self.sigmoid_prime
        elif output_func == 'relu':
            self.output_func = self.relu
        elif output_func == 'softmax':
            self.output_func  = self.softmax
            self.output_prime = self.softmax_prime

        # Fetching activation function for hidden layers
        if hidden_func == 'linear':
            self.hidden_func  = self.linear
            self.hidden_prime = self.linear_prime
        elif hidden_func == 'sigmoid':
            self.hidden_func  = self.sigmoid
            self.hidden_prime = self.sigmoid_prime
        elif hidden_func == 'relu':
            self.hidden_func = self.relu
        elif hidden_func == 'softmax':
            self.hidden_func  = self.softmax
            self.hidden_prime = self.softmax_prime


    def SSR(self, X, y, t, lam):
        """ Sum of squared residuals """
        return (t-y)**2

    def linear(self,t):
        """ Linear activation """
        a = 1
        return t*a

    def sigmoid(self,t):
        """ Sigmoid function """
        return 1 / ( 1 + np.exp(-t) )

    def relu(self, t):
        """ ReLU function """
        if t < 0:
            return 0
        else:
            return t

    def cost_derivative(self, X, y, t, theta, lam):
        """ Derivative of SSR """
        return 2*(t-y)#/self.datapoints

    def sigmoid_prime(self, t):
        return self.sigmoid(t)*( 1 - self.sigmoid(t))

    def linear_prime(self,t):
        a = 1
        return a

    def softmax(self, t):
        exponent = np.exp(t)
        return exponent/np.sum(exponent, axis=1, keepdims=True)

    def softmax_prime(self,t):
        A = self.softmax(t)
        return A*(1-A)

    def cross_entropy_gradient(self, X, y, t, theta, lam):
        return t - y
        #return -(X.T @ (y - t) - lam * theta)/self.datapoints




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
neunet = FFNN(x, y, 2, 15, cost_func='cross-entropy', hidden_func='sigmoid', output_func='softmax',
batch_size=32, eta=0.5, lam=0.000001)
#neunet.feed_forward(x)
#neunet.back_propagation()
neunet.train()
model = neunet.predict(dat.x_test, classify=True)
print(model[:20])
print(dat.y_test[:20])
I = np.where(model == dat.y_test)[0]
print(len(I) / len(dat.y_test))
