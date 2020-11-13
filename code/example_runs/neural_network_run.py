import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0,"..")
from linear_regression import Regression
from neural_network import FFNN
from load_mnist import LoadMNIST
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor, MLPClassifier
import seaborn as sns
import random
# from sklearn.metrics import accuracy_score
random.seed(1979)

"""
example run code to find best value for parameters with random
search, plot heatmap with learning rates and lambda values and compare values with sklearn
"""
def accuracy_score(Y_test, Y_pred):
    return np.sum(Y_test == Y_pred) / len(Y_test)


def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1

    return onehot_vector


def plot_heatmap():
    sns.set()
    eta_vals = np.logspace(-6, 0, 1)
    lmbd_vals = np.logspace(-6,0, 7)
    lmbd_vals = [0]
    
    train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
    test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
    
    for i in range(len(eta_vals)):
        for j in range(len(lmbd_vals)):
            neunet = FFNN(x_train, y_train, 2, 100, cost_func='cross-entropy', hidden_func='sigmoid' , output_func='softmax',
            batch_size=32, eta=eta_vals[i], lam=lmbd_vals[j])
            neunet.train()
            train_model = neunet.predict(x_train, classify=False)
            test_model = neunet.predict(x_test, classify=False)
            I = np.where(train_model == dat.y_train)[0]
            train_acc = len(I) / len(dat.y_train)
            print('train accuracy', train_acc)
            train_accuracy[i][j] = train_acc
            
            I = np.where(test_model == dat.y_test)[0]
            test_acc = len(I) / len(dat.y_test)
            print('test accuracy', test_acc)
            train_accuracy[i][j] = test_acc
    
    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="coolwarm")
    ax.set_title("Training Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.show()
    
    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="coolwarm")
    ax.set_title("Test Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.show()


def find_best_value():
    # DISABLE SEED BEFORE USING
    learning_rate = np.logspace(-3,-1,10)
    lam_vals = np.logspace(-3,-1,10)
    activation_funcs = ['sigmoid' , 'relu', 'leaky-relu'] # , 'leaky_relu'
    hidden_layers = [2,3,4,5] #hidden layer are this -1
    nodes_in_layer = [10,25,50,100,125]
    batches = [32,64]
    
    value_lst = []
    acc_lst = []
    for i in range(100):
        print (i)
        lr           = random.choice(learning_rate)
        lamb         = random.choice(lam_vals)
        activation   = random.choice(activation_funcs)
        layer = random.choice(hidden_layers)
        nodes        = random.choice(nodes_in_layer)
        batch        = random.choice(batches)
        value_lst.append([lr,lamb,activation, layer, nodes])
        print ('values', value_lst[i])
        
        neunet = FFNN(x_train, y_train, 2, 100, cost_func='cross-entropy', hidden_func='leaky-relu' , output_func='softmax',
        batch_size=32, eta=0.02, lam=0.036)
        neunet.train(epoch = 700)
        model = neunet.predict(dat.x_test, classify=True)
        I = np.where(model == dat.y_test)[0]
        accuracy = len(I) / len(dat.y_test)
        print('accuracy', accuracy)
        print("-----------------------")
    
        acc_lst.append(accuracy)
        
    index = np.argmax(np.array(acc_lst))
    print (value_lst[index])
    print (acc_lst[index])

def compare():
    # compare sklearn with own code
    NNscore = []
    sk_score = []
    for i in range(5):
        neunet = FFNN(x_train, y_train, 2, 100, cost_func='cross-entropy', hidden_func='leaky-relu' , output_func='softmax',
        batch_size=32, eta=0.036, lam=0.013)
        neunet.train(epoch = 300)
        model = neunet.predict(dat.x_test, classify=True)
        I = np.where(model == dat.y_test)[0]
        accuracy = len(I) / len(dat.y_test)
        NNscore.append(accuracy)
        print('accuracy NN', accuracy)
    
        dnn = MLPClassifier(alpha=0.013, learning_rate_init=0.036, max_iter = 300)
        dnn.fit(x_train, dat.y_train)
        score = dnn.score(x_test, dat.y_test)
        sk_score.append(score)
        print("Accuracy score on test set: ", score)
        
    print ('NN mean', np.mean(NNscore))
    print ('std NN', np.std(NNscore))
    
    print ('sk mean', np.mean(sk_score))
    print ('std sk' ,np.std(sk_score))
    
    
dat = LoadMNIST()

y_train = to_categorical_numpy(dat.y_train)
x_train = dat.x_train
y_test = to_categorical_numpy(dat.y_test)
x_test = dat.x_test

compare()