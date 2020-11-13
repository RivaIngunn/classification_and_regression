import numpy as np
import random

class FFNN:
    def __init__(self, X, y, n_layers, hidden_nodes, cost_func, hidden_func, output_func, batch_size, eta, lam=0, layerBylayer = False, layer_list = None):
        """

        Args:
            X (matrix): design matrix.
            y (matrix): target. needs to be reshaped so that it is (-1,1)
            n_layers (int): number of layers eith the exception of the input layer.
            hidden_nodes (int): number of nodes pr hidden layer.
            cost_func (string): name of cost funciton you want to use (SSR or cross-entropy).
            hidden_func (string): activation function in the hidden layers (relu, leaky-relu, sigmoid).
            output_func (string): activation function in the output layer (softmax, simoid or linear).
            batch_size (int): size of batch.
            eta (float): learning rate.
            lam (float, optional): regularization factor. Defaults to 0.
            layerBylayer (boolean, optional): changees input mode to list of layers in the network ex [20,100,100,1].
                        Defaults to False.
            layer_list (list, optional): list of layers to use for the architecture of the network 
                        if layerBylayer is True. Defaults to None.

        Returns:
            None.

        """
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
        
        #for weight initi
        self.hidden_func_name = hidden_func 
        
        #define for different way of setting up network
        self.layerBylayer = layerBylayer
        self.layer_list = layer_list
        
        # Initialize network
        self.generate_architecture()


    def generate_architecture(self):
        """
        Set up architecture of network with random weights and biases 

        Returns:
            None.

        """
        if self.layerBylayer:
            self.sizes = self.layer_list
            print(self.sizes)
        else:
            # Define structure
            self.sizes.append(self.feats)
            for l in range(self.n_layers-1):
                self.sizes.append(self.hidden_nodes)
            self.sizes.append(self.output_nodes)
            print(self.sizes)
        
        
        # Loop through layers
        for l in range(1, self.n_layers + 1):
            # Set up random layer weights and small biases
            if self.hidden_func_name == 'sigmoid':
                self.weights.append(np.random.randn( self.sizes[l-1], self.sizes[l] ) *np.sqrt(1/self.sizes[l-1]) )
            else:    
                self.weights.append(np.random.randn( self.sizes[l-1], self.sizes[l] ) *np.sqrt(2/self.sizes[l-1]) )
            
            self.bias.append(np.full( self.sizes[l], 0.01 ))

    def feed_forward(self, X):
        """
        Feed forward neural network and produce output

        Args:
            X (Matrix): design matrix of shape (n_datapoints, n_features)

        Returns:
            output of shape (n_datapoint) for regression and (n_datapoint, n_classes)
            for classification.

        """
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
        """
        Perform backpropagation

        Returns:
            None.

        """
        # Calculate output error
        #print(self.y.shape)
        #print(self.a[-1].shape)
        self.delta[-1] = self.output_prime(self.z[-1]) * self.gradient(self.X, self.y, self.a[-1], self.weights[-1], self.lam)
        # print("delta shape: ", self.delta[-1].shape)
        # Calculate errors in hidden layers
        for l in range(2, (self.n_layers + 1)):
            self.delta[-l] = (self.delta[-l + 1] @ self.weights[-l + 1].T) * self.hidden_prime(self.z[-l])
            

    
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
        """
        

        Args:
            X (matrix): design matrix of shape (n_datapoints, n_features) to use 
            for the prediction
            classify (boolean, optional): if NN is used for classification 
            classify must be set to True. Defaults to False.

        Returns:
            The predicted model

        """
        output = self.feed_forward(X)
        if classify:
            return np.argmax(output, axis=1)
        else:
            return output

    def train(self, epoch = 500):
        """
        

        Args:
            epoch (int, optional): number of epochs. Defaults to 500.

        """
        mse_lst = []
        data_indices = np.arange(self.datapoints)
        # random.seed(79)
        for i in range(epoch):
            for j in range(int(self.datapoints/self.batch_size)):
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )
                X_batch = self.X[chosen_datapoints]
                y_batch = self.y[chosen_datapoints]

                self.feed_forward(self.X)
                self.back_propagation()
            mse_lst.append(np.mean(self.delta[-1]**2))

        return mse_lst

    def fetch_funcs(self, cost_func, hidden_func, output_func):
        """
        

        Args:
            cost_func (string): name of cost function.
            hidden_func (string): name of activation function for the hidden layers.
            output_func (string): name of the activation function for the output layer.

        Returns:
            None.

        """
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
            self.hidden_prime = self.relu_prime
        elif hidden_func == 'leaky-relu':
            self.hidden_func = self.leaky_relu
            self.hidden_prime = self.leaky_relu_prime
        elif hidden_func == 'softmax':
            self.hidden_func  = self.softmax
            self.hidden_prime = self.softmax_prime
        else:
            print ("please provide a valid activation/cost function")

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
        return t * (t > 0)

    def relu_prime(self, t):
        """ derivative of relu"""
        return 1 * (t > 0)
    
    def leaky_relu(self,t, alpha = 0.1):
        """ leaky relu function"""
        a = ((t > 0) * t)                    
        b = ((t <= 0) * t * 0.01)
        return a + b
    
    def leaky_relu_prime(self,t, alpha = 0.1):
        """ derivative of leaky relu"""
        a = 1. * (t > 0)
        a[a == 0] = alpha
        return a 
        
    def cost_derivative(self, X, y, t, theta, lam):
        """ Derivative of SSR """
        return 2*(t-y)#/self.datapoints

    def sigmoid_prime(self, t):
        """ derivative of sigmoid"""
        return self.sigmoid(t)*( 1 - self.sigmoid(t))

    def linear_prime(self,t):
        """ derivative of a linear activation function"""
        a = 1
        return a

    def softmax(self, t):
        """ softmax function"""
        exponent = np.exp(t)
        return exponent/np.sum(exponent, axis=1, keepdims=True)

    def softmax_prime(self,t):
        """ derivative of softmax"""
        A = self.softmax(t)
        return A*(1-A)

    def cross_entropy_gradient(self, X, y, t, theta, lam):
        """ Derivative of cost entropy"""
        return t - y
        #return -(X.T @ (y - t) - lam * theta)/self.datapoints
