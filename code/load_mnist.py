from sklearn import datasets
from sklearn.model_selection import train_test_split

class LoadMNIST:
    def __init__(self):
        """ Load and split MNIST dataset """
        # Load MNIST dataset
        self.digits = datasets.load_digits()
        self.inputs = self.digits.data
        self.labels = self.digits.target

        # Split into training and test data
        self.x_train, self.x_test, self.y_train, self.y_test = \
        train_test_split(self.inputs, self.labels, test_size=0.2, random_state=0)

    def plot_numbers(self):
        """ plots first 5 numbers """
        plt.figure(figsize=(20,4))
        for index, (image, label) in enumerate(zip(self.inputs[0:5], self.labels[0:5])):
            plt.subplot(1, 5, index + 1)
            plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
            plt.title('Training: %i\n' % label, fontsize = 20)
        plt.show()
