import numpy as np


def get_im2col_indices(x_shape, field_height=3, field_width=3, padding=1, stride=1):
    # figure out what the size of the output should be
    N, C, H, W = x_shape
    out_height = (H + 2 * padding - field_height) / stride + 1
    out_width = (W + 2 * padding - field_width) / stride + 1

    i0 = np.repeat(np.arange(field_height,dtype='int32'), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height,dtype='int32'), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width,dtype='int32'), int(out_height))
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C,dtype='int32'), field_height * field_width).reshape(-1, 1)

    return (k, i, j)

class Conv():
    def __init__(self, n_filter, h_filter, w_filter, stride, padding, input_shape):

        self.channels, self.height, self.width = input_shape

        self.n_filter = n_filter
        self.h_filter = h_filter
        self.w_filter = w_filter
        
        self.stride = stride
        self.padding = padding

        self.W = np.random.randn(n_filter, self.channels, h_filter, w_filter) / np.sqrt(n_filter / 2.)
        self.b = np.zeros((self.n_filter, 1))
        self.params = [self.W, self.b]

        self.h_out = int((self.height - h_filter + 2 * padding) / stride + 1)
        self.w_out = int((self.width - w_filter + 2 * padding) / stride + 1)

    def forward(self, X):
        self.n_X = X.shape[0]
        
        x_padded = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        k, i, j = get_im2col_indices(X.shape, self.h_filter, self.w_filter, self.padding, self.stride)
        cols = x_padded[:, k, i, j]
        self.X_col = cols.transpose(1, 2, 0).reshape(self.h_filter * self.w_filter * X.shape[1], -1)

        W_row = self.W.reshape(self.n_filter, -1)

        out = W_row @ self.X_col + self.b

        out = out.reshape(self.n_filter, self.h_out, self.w_out, self.n_X)
        out = out.transpose(3, 0, 1, 2)

        self.out = np.maximum(0, out)
        
        return self.out


    def backward(self, dout):
        # relu
        dout[self.out <= 0] = 0

        dout_flat = dout.transpose(1, 2, 3, 0).reshape(self.n_filter, -1)

        dW = dout_flat @ self.X_col.T
        dW = dW.reshape(self.W.shape)

        db = np.sum(dout, axis=(0, 2, 3)).reshape(self.n_filter, -1)

        W_flat = self.W.reshape(self.n_filter, -1)

        cols = W_flat.T @ dout_flat
        shape = (self.n_X, self.channels, self.height, self.width)

        N, C, H, W = shape
        H_padded, W_padded = H + 2 * self.padding, W + 2 * self.padding
        x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
        k, i, j = get_im2col_indices(shape, self.h_filter, self.w_filter, self.padding, self.stride)
        cols_reshaped = cols.reshape(C * self.h_filter * self.w_filter, -1, N)
        cols_reshaped = cols_reshaped.transpose(2, 0, 1)
        np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)

        dX = x_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]

        return dX, [dW, db]


class Maxpool():
    def __init__(self, X_dim, size, stride):

        self.channels, self.height, self.width = X_dim

        self.params = []

        self.size = size
        self.stride = stride

        self.h_out = (self.height - size) / stride + 1
        self.w_out = (self.width - size) / stride + 1

        self.h_out, self.w_out = int(self.h_out), int(self.w_out)
        self.out_dim = (self.channels, self.h_out, self.w_out)

    def forward(self, X):
        self.n_X = X.shape[0]
        X_reshaped = X.reshape(X.shape[0] * X.shape[1], 1, X.shape[2], X.shape[3])

        padding = 0
        x_padded = np.pad(X_reshaped, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
        k, i, j = get_im2col_indices(X_reshaped.shape, self.size, self.size, padding, self.stride)
        cols = x_padded[:, k, i, j]
        self.X_col = cols.transpose(1, 2, 0).reshape(self.size * self.size * X_reshaped.shape[1], -1)

        self.max_indexes = np.argmax(self.X_col, axis=0)
        out = self.X_col[self.max_indexes, range(self.max_indexes.size)]

        out = out.reshape(self.h_out, self.w_out, self.n_X,
                          self.channels).transpose(2, 3, 0, 1)
        return out

    def backward(self, dout):
        dX_col = np.zeros_like(self.X_col)
        # flatten the gradient
        dout_flat = dout.transpose(2, 3, 0, 1).ravel()

        dX_col[self.max_indexes, range(self.max_indexes.size)] = dout_flat

        # get the original X_reshaped structure from col2im
        shape = (self.n_X * self.channels, 1, self.height, self.width)

        N, C, H, W = shape
        dX = np.zeros((N, C, H, W), dtype=dX_col.dtype)
        k, i, j = get_im2col_indices(shape, self.size, self.size, 0, self.stride)
        cols_reshaped = dX_col.reshape(C * self.size * self.size, -1, N)
        cols_reshaped = cols_reshaped.transpose(2, 0, 1)
        np.add.at(dX, (slice(None), k, i, j), cols_reshaped)


        dX = dX.reshape(self.n_X, self.channels, self.height, self.width)
        return dX, []


class Flatten():
    def __init__(self):
        self.params = []

    def forward(self, X):
        self.X_shape = X.shape
        self.out_shape = (self.X_shape[0], -1)
        out = X.ravel().reshape(self.out_shape)
        self.out_shape = self.out_shape[1]
        return out

    def backward(self, dout):
        out = dout.reshape(self.X_shape)
        return out, ()


class FullyConnected():
    def __init__(self, in_size, out_size):

        self.W = np.random.randn(in_size, out_size) / np.sqrt(in_size / 2.)
        self.b = np.zeros((1, out_size))
        self.params = [self.W, self.b]

    def forward(self, X):
        self.X = X
        out = self.X @ self.W + self.b
        return out

    def backward(self, dout):
        dW = self.X.T @ dout
        db = np.sum(dout, axis=0)
        dX = dout @ self.W.T
        return dX, [dW, db]
