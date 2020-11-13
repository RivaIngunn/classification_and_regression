#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from neural_network import FFNN
from linear_regression import Regression
import unittest

class TestNeuralNetwork(unittest.TestCase):

        
    def test_architecture(self):
        reg = Regression()
        reg.dataset_franke(10)
        reg.design_matrix(1)
        reg.split(reg.X, reg.f)
        f_train = reg.f_train.reshape(-1,1)
        nodes = 5
        layers = 3 # 2 hidden layers
        
        
        
        NN = FFNN(reg.X_train, f_train, layers, nodes, cost_func='SSR', hidden_func='leaky-relu' , output_func='linear',
        batch_size=32, eta=0.1, lam=0.1)
        
        expected_size = [2,5,5,1]
        actual_size = NN.sizes
        #test that the correct architecture is made from the input
        self.assertEqual(actual_size, expected_size)
        
        expected_w = [(2,5),(5,5),(5,1)]
        expected_b = [(5,),(5,),(1,)]

        # test that the weights and biases are the correct shape
        for i in range(layers):
            self.assertEqual(NN.weights[i].shape, expected_w[i])
            self.assertEqual(NN.bias[i].shape, expected_b[i])
        
    def test_delta(self):
        reg = Regression()
        reg.dataset_franke(10)
        reg.design_matrix(1)
        reg.split(reg.X, reg.f)
        f_train = reg.f_train.reshape(-1,1)
        nodes = 5
        layers = 3 # 2 hidden layers
        
        NN = FFNN(reg.X_train, f_train, layers, nodes, cost_func='SSR', hidden_func='leaky-relu' , output_func='linear',
        batch_size=8, eta=0.1, lam=0.1)
        NN.train(epoch = 2)
        
        #test shape of delta
        expected_delta = [(8,5),(8,5),(8,1)]
        for i in range (layers):
            self.assertEqual(NN.delta[i].shape, expected_delta[i])
       
        
if __name__ == '__main__':

    unittest.main()