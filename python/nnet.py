import numpy as np

def SoftmaxLoss(X, y):
    dx = softmax(X)
    dx[range(y.shape[0]), y] -= 1
    dx /= y.shape[0]
    return dx

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class CNN:
    def __init__(self, layers):
        self.layers = layers
        self.params = []
        for layer in self.layers:
            self.params.append(layer.params)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dout):
        grads = []
        for layer in reversed(self.layers):
            dout, grad = layer.backward(dout)
            grads.append(grad)
        return grads

    def train_step(self, X, y):
        out = self.forward(X)
        dout = SoftmaxLoss(out, y)
        grads = self.backward(dout)

        return grads

    def predict(self, X):
        X = self.forward(X)
        return np.argmax(softmax(X), axis=1)