#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

#seed
np.random.seed(42)

class FF_NeuralNetwork:
    def __init__(self, input_values):
      
        self.weights = []
        self.bias = []
        
        self.nabla_bias = []
        self.nabla_weights = []

        self.z = []
        
        self.activations = []
        self.activations.append(np.array(input_values))
    
    def create_layers(self, architecture):
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
    
        
    def feed_forward(self, act_func):
        for i, weight in enumerate(self.weights):
            
            # print ('weight',self.weights[i].shape)
            # print ('activations', np.array(self.activations[i]).shape)
            # print ('bias',self.bias[i].shape)
            
            
            self.z.append(np.array(np.dot( weight, self.activations[i]) + self.bias[i]))
            
            # print ('z',self.z[i].shape)
            self.activations.append(np.array(act_func(self.z[i])))
            # print ('active',self.activations[i+1].shape)

    def back_prpagation(self,act_func_deriv):
        print ('acitive' , len(self.activations))
        print ('zzz', len(self.z))
        
        for i in reversed(range(len(self.activations)-1)):
            l=i+1
            print ('i = ', i)
            print (len(self.z))
            # no z value for the first layer so we have to use index [l-1]
            da_dz = act_func_deriv(self.z[l-1])
            #maybe wron index
            dz_dw = self.activations[l]
            
            
            dc_da = 'autograd'

        # # want to find dc_dw    
        # dc_dw = dc_da*da_dz*dz_dw
        # #dc_db
        # dc_db = da_dz*dc_da
        
        
        # delta = da_dz * dc_da
        
    
        # #updating weights
        # weights = weights -lr*dc_dw
        # #updating bias
        # for element in deriv:
        #     bias -= lr*dc_db
                
        
        return 

        
arch = [2,4,4,2]
FFNN = FF_NeuralNetwork([1,0])
FFNN.create_layers(arch)
FFNN.feed_forward(FFNN.sigmoid)
FFNN.back_prpagation(FFNN.sigmoid_deriv)
# for val in FFNN.weights:
#     print (val)
#print (FFNN.bias)


# print (FFNN.weights[1])
