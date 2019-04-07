import numpy as np
from layers import *
from nnet import CNN
from sklearn.utils import shuffle


def load_mnist(test_size=0.2):
    X, y = [], []
    for line in open('data/mnist_test.csv'):
        values = line.strip('\n').split(',')
        y.append(values[0])
        X.append([values[1:]])

    shape = (-1, 1, 28, 28)
    X = np.array(X, dtype="int32").reshape(shape)
    y = np.array(y, dtype="int32")

    X, y = shuffle(X, y)

    idx = int(len(y) * (1 - test_size))
    X_train, X_test = X[:idx], X[idx:]
    y_train, y_test = y[:idx], y[idx:]

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist()

    print('Training samples:', X_train.shape, y_train.shape)
    print('Testing samples', X_test.shape, y_test.shape)

    conv = Conv(32, h_filter=3, w_filter=3, stride=1, padding=1, input_shape=(1, 28, 28))
    maxpool = Maxpool((32, 28, 28), size=2, stride=1)
    flat = Flatten()
    fc = FullyConnected(np.prod(maxpool.out_dim), 10)
    
    cnn = CNN([conv, maxpool, flat, fc])

    learning_rate = 0.01
    for e in range(10):

        grads = cnn.train_step(X_train[:100], y_train[:100])

        for param, grad in zip(cnn.params, reversed(grads)):
            for i in range(len(grad)):
                param[i] += - learning_rate * grad[i]

        train_acc = np.mean(cnn.predict(X_train[:100]) == y_train[:100])
        test_acc = np.mean(cnn.predict(X_test[:25]) == y_test[:25])
        print("\nEpoch {2}\n========\nTraining Accuracy = {0} | Test Accuracy = {1}".format(train_acc, test_acc, e))
