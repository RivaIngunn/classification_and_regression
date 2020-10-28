import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Import classes
import sys
sys.path.insert(0,"..")
from linear_regression import Regression
from stochastic_gradient_descent import stochastic_descent
from logistic_regression import LogisticRegressionOwn

# Load MNIST dataset
digits = datasets.load_digits()
inputs = digits.data
labels = digits.target
print (inputs.shape)

# Plot first 5 numbers with their labels
plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(inputs[0:5], labels[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize = 20)
plt.show()

# Split into training and test data
x_train, x_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, random_state=0)
"""
# Logistic regression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
pred = logreg.predict(x_test)

# Checking accuracy
score = logreg.score(x_test, y_test)
print(score)


print("Running own logistich regression!")
logreg2 = LogisticRegressionOwn()

logreg2.fit(x_train, y_train, 0)
print(x_train.shape)
print(y_train.shape)

pred2 = logreg2.predict(x_test)
print(logreg2.beta.shape)
print(pred2[:20])
print(y_test[:20])
"""
def softmax(z):
    output = np.zeros_like(O)
    for i in range(z.shape[0]):
        top = np.exp(z[i])
        bottom = np.sum(top)
        output[i] = top/bottom
    return output

n_dat = x_train.shape[0]
feats = x_train.shape[1]
n_class = 10
w = np.random.rand(feats,n_class)
O = x_train @ w
model = softmax(O)
print(np.max(model))
print(model.shape)
print(y_train.shape)
def cross_entropy(target, model):
    return -(np.log(model) @ target)

print(cross_entropy(y_train, softmax(O)))
