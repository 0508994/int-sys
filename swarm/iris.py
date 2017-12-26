import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

class IrisDF:
    def __init__(self, normalize=True, test_size=0.3):
        iris = datasets.load_iris()
        X = None
        if normalize:
            X = preprocessing.normalize(iris.data) 
        else:
            X = iris.data
        y = self.encode(iris.target)

        self.X_train, self.X_test, self.y_train, self.y_test = \
                train_test_split(X, y, test_size = test_size)

    def encode(self, data):
        # encodes the target label y so it's compatible with nn output layer
        y = []
        for i in data:
            if i == 0:
                y.append([0, 0, 1])
            elif i == 1:
                y.append([0, 1, 0])
            else:
                y.append([1, 0, 0])
        return np.array(y)

    def subset(self, train=24, test=6):
        # reshapes the dataset into a smaller subset
        if train > len(self.X_train) or \
            test > len(self.X_test):
            return
        self.X_train = self.X_train[:train]
        self.y_train = self.y_train[:train]
        self.X_test = self.X_test[:test]
        self.y_test = self.y_test[:test]
        


class IrisNN:
    def __init__(self):
        # define Hiperparameters
        self.input_size = 4
        self.hidden_size = 6
        self.output_size = 3

        # init weights with random values
        self.W1 = np.random.rand(self.input_size, self.hidden_size)
        self.W2 = np.random.rand(self.hidden_size, self.output_size)

        # init biases with random values
        self.b1 = np.random.rand(self.input_size, self.hidden_size)
        self.b2 = np.random.rand(self.hidden_size, self.output_size)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
    def softmax(self, z):
        assert len(z.shape) == 2
        s = np.max(z, axis=1)
        s = s[:, np.newaxis]
        e_x = np.exp(z - s)
        div = np.sum(e_x, axis=1)
        div = div[:, np.newaxis]
        return e_x / div
    
    def forward(self, X):
        # Propagate inputs through the network
        z2 = np.dot(X, self.W1) + self.b1
        a2 = self.sigmoid(z2)
        z3 = np.dot(a2, self.W2) + self.b2
        self.y_hat = self.softmax(z3)

    def mse(self, y):
        return
