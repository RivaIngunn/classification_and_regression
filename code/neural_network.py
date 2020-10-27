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
    def __init__(self, input_values,y):
        self.input_values = input_values
        self.y = y
        self.weights = []
        self.bias = []
        
        self.nabla_bias = []
        self.nabla_weights = []

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
            self.bias.append(np.array(bias))
            
        self.weights= self.weights[1:]
        self.bias = self.bias[1:]
        

    
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
 
    def sigmoid_deriv(self,x):
        return self.sigmoid(x) * (1-self.sigmoid(x))
    
    def MSE(X, beta, y):
        return np.mean(( X @ beta - y)**2)
    
    
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
        errror[-1] = self.activations[-1] -y
        
        weights_gradient[-1] = self.activations[-1].T @ error[-1]
        bias gradient[-1] = np.sum(error[-1],axis = 0)
        
        # print ('acitive' , self.activations)
        # print ('zzz', len(self.z))
        
        # #values for layer L (last layer)
        # dc_da = 2*(self.activations[-1]-self.y)
        # da_dz = act_func_deriv(self.z[-1])
        # print ('dc_da', dc_da)
        # print ('da_dz', da_dz)
        # delta = dc_da * da_dz
        # weight_gradient[-1] = self.activations[-1].T @
        
        for l in reversed(range(len(self.activations)-1)):
            #l=L-1 for last layer
            
        error[l] = (error[l+1] @ self.weights[l].T ) *self.activations[l-1] * (1-self.activations[l-1])
        weights_gradient[l] = ?
        bias_gradienet[l] =  np.sum(error[l], axis = 0)
        
        #     # no z or weights value for the first layer so we have to use index [l-1]

        #     print ('l = ', l)
        #     print ('len z' , len(self.z))
        #     print ('len weights', len(self.weights))
            
        #     # calculatin derivatives.
            
        #     # no z value for the first layer so we have to use index [l-1]
        #     da_dz = act_func_deriv(self.z[l-1])
            
        #     #maybe wrong index
        #     dz_dw = self.activations[l]
        #     print ('activations ', self.activations[l])
            
        #     print ('...........................................')
        #     print ('\n delta \n', delta )
        #     print ('\n weights \n', self.weights[l])
        #     print ('\n act_func_deriv \n', act_func_deriv(self.z[l-1]).shape)
        #     print ('res', self.weights[l] @ self.z[l-1])
        #     print ('...........................................')
            
            
        #     delta = delta @ self.weights[l] @ act_func_deriv(self.z[l-1])

        #     weights_gradient = self.activations[l]
            # dc_dw = delta*dz_dw
            # dc_db = delta *1 

            # #updating weights
            # weights = weights -lr*dc_dw
            # #updating bias
            # for element in deriv:
            #     bias -= lr*dc_db
    #regularization
    
    #for i in range(1000)
    weights -=  lr*weights_gradient
    bias -= lr*bias_gradient
        return 

        

FFNN = FF_NeuralNetwork([1,0], [1,1])
FFNN.create_layers(4,5,1)
FFNN.feed_forward(FFNN.sigmoid)
FFNN.back_prpagation(FFNN.sigmoid_deriv)
# for val in FFNN.weights:
#     print (val)
#print (FFNN.bias)


# print (FFNN.weights[1])
