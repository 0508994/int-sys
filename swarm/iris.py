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
    def __init__(self, gen_randomly=False):
        # define Hiperparameters
        self.input_size = 4
        self.hidden_size = 6
        self.output_size = 3

        if gen_randomly:
            # init weights with random values
            self.W1 = np.random.rand(self.input_size, self.hidden_size)
            self.W2 = np.random.rand(self.hidden_size, self.output_size)
            # init biases with random values
            self.b1 = np.random.rand(self.hidden_size)
            self.b2 = np.random.rand(self.output_size)

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

    def compute_mse(self, X_train, y_train):
        # mse_sum = 0
        # for i, j in zip(y, self.y_hat):
        #     mse_sum += sum((j - i)**2)
        # return mse_sum / float(len(y))
        self.forward(X_train)
        return sum(sum((y_train - self.y_hat)**2)) / len(y_train)

    def compute_accuracy(self, X_test, y_test):
        self.forward(X_test)
        num_correct = 0
        for i, j in zip(y_test, self.y_hat):
            if i.argmax() == j.argmax():
                num_correct += 1
        return float(num_correct) / len(y_test)

    def merge_params(self):
        # used in PSO as particle position
        return np.concatenate([self.W1.flatten(), self.b1,\
               self.W2.flatten(), self.b2], axis=0) 

    def unpack_params(self, data):
        offset = self.input_size * self.hidden_size
        self.W1 = data[:offset].reshape(self.input_size, self.hidden_size)
        self.b1 = data[offset:offset + self.hidden_size]
        offset += self.hidden_size
        self.W2 = data[offset:offset + self.hidden_size * self.output_size]\
                               .reshape(self.hidden_size, self.output_size)
        offset += self.hidden_size * self.output_size
        self.b2 = data[offset:]