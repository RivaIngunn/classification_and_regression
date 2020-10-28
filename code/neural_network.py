#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 17:56:43 2020

@author: ingunn
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt

#seed
np.random.seed(42)

class FF_NeuralNetwork:
    def __init__(self, input_values,y, lam=0):
        self.input_values = input_values
        self.y = y
        self.lam = lam
        self.weights = []
        self.bias = []

        self.bias_gradient = []
        self.weights_gradient = []
        self.error = []
        
        
        self.z = []
        
        self.activations = []
        self.activations.append(np.array(self.input_values)) # ? 
    
    def create_layers(self, nodes_hidden ,layers_hidden, nodes_output):

        architecture = []
        architecture.append(len(self.input_values))
        for i in range(layers_hidden):
            architecture.append(nodes_hidden)
        architecture.append(nodes_output)
        
        for i, size in enumerate(architecture):
            lst = []
            bias = []
            for j in range(size):
                lst.append(np.zeros(architecture[i-1]))
                bias.append(0)
            self.weights.append(np.array(lst))
            self.weights_gradient.append(np.zeros_like(lst))
            
            self.bias.append(np.array(bias))
            self.bias_gradient.append(np.zeros_like(bias))
            self.error.append(np.zeros_like(bias))
            
        self.weights= self.weights[1:]
        self.bias = self.bias[1:]
        

    
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
 
    def sigmoid_deriv(self,x):
        return self.sigmoid(x) * (1-self.sigmoid(x))
    
    def MSE(X, beta, y):
        return np.mean(( X @ beta - y)**2)
    
    def cost_func_regularized(func,x):
        value = 0
        for i in range(len(self.weights)):
            sum_of_weights += np.sum(self.weights[i])
            
        return func(x) + self.lam* sum_of_weights
        
    def feed_forward(self, act_func):
        for i, weight in enumerate(self.weights):
            
            # print ('weight',self.weights[i].shape)
            # print ('activations', np.array(self.activations[i]).shape)
            # print ('bias',self.bias[i].shape)
            
            
            self.z.append(np.array(np.dot( weight, self.activations[i]) + self.bias[i]))
            
            # print ('z',self.z[i].shape)
            self.activations.append(np.array(act_func(self.z[i])))
            # print ('active',self.activations[i+1].shape)
            
            # make a new activation fuction at the end

    def back_prpagation(self,act_func_deriv):
        
        
        
        
        
        
        self.error[-1] = self.activations[-1] - self.y
        
        print ("-------------------------")
        print ('y',len(self.y))
        print ('activ',len(self.activations))
        print ('weight',len(self.weights))
        print("---------")
        print ('activ last',self.activations[-1].shape)
        print ('weight last',self.weights[-1].shape)
        print ('error', self.error[-1].shape)
        print ("--------------------------")
        
        
        self.weights_gradient[-1] = self.activations[-1].T @ self.error[-1]
        self.bias_gradient[-1] = np.sum(self.error[-1],axis = 0)
        
        
        
        
        # print ('acitive' , self.activations)
        # print ('zzz', len(self.z))
        
        # #values for layer L (last layer)
        dc_da = 2*(self.activations[-1]-self.y)
        da_dz = act_func_deriv(self.z[-1])
        # print ('dc_da', dc_da)
        # print ('da_dz', da_dz)
        delta = dc_da * da_dz
        # weight_gradient[-1] = self.activations[-1].T @
        
        for l in reversed(range(len(self.activations)-1)):
            #l=L-1 for last layer
            
            print ('...........................................')
            print ('l', l)
            print ('zzzzzzzzzz', len(self.z))
            print ('error[l+1]', self.error[l+1].shape)
            print ('weights', self.weights[l].shape)
            print ('activatoins[l-1]')
            print ('activations', self.activations[l].shape)
            print ('res', self.weights[l] @ self.z[l-1])
            print ('...........................................')
        
            
            #delta = np.sum(delta*self.weights[l] @ self.activations[self.z[l]])
            self.error[l] = (self.error[l+1] @ self.weights[l]) *self.activations[l] * (1-self.activations[l])
            #weights_gradient[l] = ?
            self.bias_gradient[l] =  np.sum(self.error[l], axis = 0)
        
            # no z or weights value for the first layer so we have to use index [l-1]

 
    def update_weight_bias(self,n):
        for i in range(n):
            self.weights = self.learning_rate* self.weights_gradient
            self.bias = self.learning_rate* self.bias_gradient

FFNN = FF_NeuralNetwork([1,0], [1,1])
FFNN.create_layers(4,5,2)
FFNN.feed_forward(FFNN.sigmoid)
FFNN.back_prpagation(FFNN.sigmoid_deriv)
# for val in FFNN.weights:
#     print (val)
#print (FFNN.bias)


# print (FFNN.weights[1])
