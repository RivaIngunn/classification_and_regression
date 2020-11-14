#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural network example run on regression with the franke function. 
inclues a random search to find best parameters, an R2 plot
for learning rate /lambda and comparison with sklearn
"""


import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0,"..")
from linear_regression import Regression
from neural_network import FFNN
from load_mnist import LoadMNIST
import tensorflow as tf
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error
import seaborn as sns
from pprint import pprint
import random
from sklearn.model_selection import train_test_split

random.seed(1979)
def NN_with_sklearn(X_train,X_test,y_train, y_test,rate, lam, hidden_nodes_inlayer, epochs):
    

    dnn = MLPRegressor(hidden_layer_sizes=(hidden_nodes_inlayer),
                        solver = 'sgd', alpha=lam, learning_rate_init=rate,batch_size = 32, max_iter=epochs)
    dnn.fit(X_train, y_train)
    print("SKLEARN Accuracy score on test set: ", dnn.score(X_test, y_test))

        
def R2(y,y_pred):
        top = np.mean( (y - y_pred)**2 )
        bot = np.mean( (y - np.mean(y))**2 )
        return 1 - top/bot


def find_best_value():
    #DISABLE SEED BEFORE USING
    learning_rate = np.logspace(-4,-1,30)
    lam_vals = np.logspace(-4,-1,30)
    activation_funcs = ['sigmoid' , 'leaky-relu'] 
    hidden_layers = [2] #hidden layer are this -1
    nodes_in_layer = [32,64,70,100]
    batches = [32]
    
    
    acc_lst = []
    r2_lst = []
    value_lst = []
    for i in range(1000):
    
        print (i)
        lr           = random.choice(learning_rate)
        lamb         = random.choice(lam_vals)
        activation   = random.choice(activation_funcs)
        layer        = random.choice(hidden_layers)
        nodes        = random.choice(nodes_in_layer)
        batch        = random.choice(batches)
        value_lst.append([lr,lamb,activation, layer, nodes, batch])
        print ('values', value_lst[i])
        
        neunet = FFNN(reg.X_train, y_train, layer, nodes, cost_func='SSR', hidden_func= activation , output_func='linear',
        batch_size=batch, eta=lr, lam=lamb)
        neunet.train(epoch = 500)
        model = neunet.predict(reg.X_test, classify=False)
        mse = np.mean((y_test - model)**2)
        r2 = R2(y_test,model)
        r2_lst.append(r2)
        print ('r2' , r2)
        print("-----------------------")
      

    
    r2_lst = np.array(r2_lst)
    value_lst = np.array(value_lst)
    
    r2_lst[np.isnan(r2_lst)] = 0
    index = np.argmax(r2_lst)
    print (value_lst[index])
    print ('r2' , r2_lst[index])



def plot_heatmap():
    sns.set()
    eta_vals = (np.logspace(-6,-1,10))
    lmbd_vals = (np.logspace(-6,1,10))
    train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
    test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
    
    for i in range(len(eta_vals)):
        for j in range(len(lmbd_vals)):
            neunet = FFNN(reg.X_train, y_train, 2, 100, cost_func='SSR', hidden_func='leaky-relu' , output_func='linear',
            batch_size=32, eta=eta_vals[i], lam=lmbd_vals[j])
            neunet.train(epoch = 300)
            train_model = neunet.predict(reg.X_train, classify=False)
            test_model = neunet.predict(reg.X_test, classify=False)
            train_accuracy[i][j] = R2(y_train,train_model)
            test_accuracy[i][j] = R2(y_test,test_model)
            print (R2(y_test,test_model))
            
            
            # # --------- WITH SKLARN HEATMAP
            # dnn = MLPRegressor(alpha=lmbd_vals[j], learning_rate_init=eta_vals[i], max_iter=200)
            # dnn.fit(reg.X_train, reg.f_train)
            # model = dnn.predict(reg.X_test)
            # test_accuracy[i][j] = R2(reg.f_test, model)
            
    
    ticks_x = [ '1.e-06', '1.e-05', '1.e-04', '1.e-03', '1.e-02', '1.e-01','1.e+00' ,'1.e+01' ]
    ticks_y = [ '1.e-06', '1.e-05', '1.e-04', '1.e-03', '1.e-02', '1.e-01']
    test_accuracy[np.where(test_accuracy <0)] = 0 
    train_accuracy[np.where(train_accuracy <0)] = 0 
    
    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(train_accuracy, cmap="coolwarm", xticklabels = ticks_x, yticklabels = ticks_y).invert_yaxis()
    ax.set_title("$R^2$ TRAIN")
    ax.set_ylabel("learning rate: ($\eta$)" , fontsize='16')
    ax.set_xlabel(r' regularization factor: $(\lambda)$', fontsize='16')
    plt.show()
    
    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(test_accuracy,annot = True, cmap="coolwarm", xticklabels = ticks_x, yticklabels = ticks_y).invert_yaxis()
    ax.set_title("$R^2$ score as function of learning rate and penalty",fontsize='17')
    ax.set_ylabel('learning rate($\eta$)' , fontsize='17')
    ax.set_xlabel(r' regularization factor($\lambda)$', fontsize='17')
    plt.savefig('../../visuals/NN_lambdaLearningRate_test.pdf')
    plt.show()


def compare():
    # compare with  sklearn and OLS
    ols_score = []
    NNscore = []
    sk_score = []
    value = [0.01,0.015,0.02]
    for i in range(1):
        neunet = FFNN(reg.X_train, y_train, 2, 100, cost_func='SSR', hidden_func='sigmoid' , output_func='linear',
        batch_size=32, eta=0.008, lam=0.005)
        neunet.train(epoch = 700)
        model = neunet.predict(reg.X_test, classify=False)
        r2 = R2(y_test,model)
        NNscore.append(r2)
        print ('NN R2' , r2)
    
        
        f_tilde, f_pred = reg.OLS(reg.X_train, reg.X_test, reg.f_train)
        # Calculate errors
        r2_OLS = R2(reg.f_test, f_pred)
        ols_score.append(r2_OLS)
        print ('R2 score OLS:', r2_OLS)
        
        dnn = MLPRegressor(max_iter=1000,hidden_layer_sizes=(100))#hidden_layer_sizes=100,activation = 'logistic',
                            #solver = 'sgd', alpha=0.00026, learning_rate_init=0.0853,batch_size = 32, max_iter=1000)
        dnn.fit(reg.X_train, reg.f_train)
        score = dnn.score(reg.X_test, reg.f_test)
        sk_score.append(score)
        print("SKLEARN Accuracy score on test set: ", score)
        
    print ('NN mean', np.mean(NNscore))
    print ('std NN', np.std(NNscore))
    
    print ('sk mean', np.mean(sk_score))
    print ('std sk' ,np.std(sk_score))
    
    print ('OLS mean', np.mean(ols_score))
    print ('std OLS' ,np.std(ols_score))
    


def convex_check():
    plt.style.use('seaborn-whitegrid')
    epoch_n = 1000
    epoch = np.linspace(0,epoch_n,epoch_n)
    act_func = ['sigmoid', 'relu', 'leaky-relu']
    lr_lst= [95,100,105]
    for i in range(1):
        neunet = FFNN(reg.X_train, y_train, 2,100, cost_func='SSR', hidden_func='leaky-relu' , output_func='linear',
        batch_size=32, eta=0.00788, lam=0.0057)
        loss = neunet.train(epoch = epoch_n)
        plt.plot(epoch[500:], loss[500:], label = lr_lst[i])
    
    plt.xlabel('Epochs', fontsize = '16')
    plt.ylabel('Training MSE', fontsize = '16')
    plt.title('Comparison of convergence for learning rate',fontsize = '16')
    plt.legend()
    
    plt.savefig('../../visuals/convergence_comparison.pdf')
    plt.show()



def compare_activation():
    plt.style.use('seaborn-whitegrid')
    epoch_n = 800
    epoch = np.linspace(0,epoch_n,epoch_n)
    act_func = ['sigmoid', 'leaky-relu']
    
    for i in range(1):
        neunet = FFNN(reg.X_train, y_train, 2, 100, cost_func='SSR', hidden_func=act_func[i] , output_func='linear',
        batch_size=32, eta=0.008, lam=0.005)
        loss = neunet.train(epoch = epoch_n)
        plt.plot(epoch, loss, label = act_func[i] )
    
    plt.xlabel('Epochs', fontsize = '16')
    plt.ylabel('Training MSE', fontsize = '16')
    plt.title('Comparison of activation functions',fontsize = '16')
    plt.legend()
    
    plt.savefig('../../visuals/activation_function_compare.pdf')
    plt.show()
    
def best_value_sk():
    mlp = MLPRegressor(max_iter=1000)
    parameter_space = {
        'hidden_layer_sizes': [(10,30,10),(100,), (36,36), (10,10,10),(200)],
        'activation': ['tanh', 'relu','logistic'],
        'solver': ['sgd', 'adam'],
        'alpha': np.logspace(-6,-1,10),
        'learning_rate': ['constant','adaptive'],
    }
    from sklearn.model_selection import RandomizedSearchCV
    clf = RandomizedSearchCV(mlp, parameter_space, n_iter = 100)
    clf.fit(reg.X_train, reg.f_train) # X is train samples and y is the corresponding labels
    print('Best parameters found:\n', clf.best_params_)


    
reg = Regression()
reg.dataset_franke(1500)
reg.design_matrix(5)
reg.split(reg.X, reg.f)

y_train = reg.f_train.reshape(-1,1)
y_test = reg.f_test.reshape(-1,1)

#compare OLS sklearn and our NN
compare()

# check if values converge towards a defferent value
# convex_check()

# compare convergence for activation functions
# compare_activation()

# tries to find the best value through random grid search
# find_best_value()

# best value with sk
# best_value_sk()


dnn = MLPRegressor(max_iter=1000,hidden_layer_sizes=(36,36),activation = 'relu',
                    solver = 'adam', alpha= 0.007742636826811277)
dnn.fit(reg.X_train, reg.f_train)
score = dnn.score(reg.X_test, reg.f_test)
print("SKLEARN Accuracy score on test set: ", score)