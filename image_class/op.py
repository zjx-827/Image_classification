import os.path

import numpy as np
from activate import ReLU

np.random.seed(1005)


class Op(object):
    def __init__(self):
        self.inputs = None

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grads):
        raise NotImplementedError


class Linear(Op):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.params = {'W': np.random.normal(0, 1, size=(out_features, in_features)),
                       'b': np.zeros(shape=out_features)}
        self.grads = {'W': np.zeros(shape=(out_features, in_features)), 'b': np.zeros(shape=out_features)}

    def forward(self, inputs):
        """
        :param inputs: shape = [batch_size, in_features]
        :return: y_pred: shape = [batch_size, out_features]
        """
        self.inputs = inputs.astype(np.float32)
        y_pred = np.dot(inputs, self.params['W'].T) + self.params['b']
        return y_pred

    def backward(self, grads_out):
        """
        :param grads_out: shape = [batch_size, out_features]
        :return: grads_in: shape = [batch_size, in_features]
        """
        # dJ/dW = sum(Δy*x), dJ/db = sum(Δy)
        self.grads['W'] = np.dot(grads_out.T, self.inputs)  # [out_features, batch_size] * [batch_size, in_features]
        self.grads['b'] = np.sum(grads_out, axis=0)  # [batch_size,]
        # print("***", self.grads['W'])
        # print("***", self.grads['b'])
        return np.dot(grads_out, self.params['W'])  # [batch_size, out_features] * [out_features, in_features]


class FCNet3(Op):
    def __init__(self, in_features, hidden_size1, hidden_size2, out_features):
        super(FCNet3, self).__init__()
        self.fc1 = Linear(in_features, hidden_size1)
        self.act1 = ReLU()
        self.fc2 = Linear(hidden_size1, hidden_size2)
        self.act2 = ReLU()
        self.fc3 = Linear(hidden_size2, out_features)
        self.fc_layers = [self.fc1, self.fc2, self.fc3]

    def forward(self, inputs):
        z1 = self.fc1(inputs)
        a1 = self.act1(z1)
        z2 = self.fc2(a1)
        a2 = self.act2(z2)
        z3 = self.fc3(a2)
        return z3

    def backward(self, loss_grad_z3):
        loss_grad_a2 = self.fc3.backward(loss_grad_z3)
        loss_grad_z2 = self.act2.backward(loss_grad_a2)
        loss_grad_a1 = self.fc2.backward(loss_grad_z2)
        loss_grad_z1 = self.act1.backward(loss_grad_a1)
        self.fc1.backward(loss_grad_z1)

    def get_weights(self):
        parameters = []
        for layer in self.fc_layers:
            parameters.append(layer.params['W'])
        return parameters

    def get_biases(self):
        parameters = []
        for layer in self.fc_layers:
            parameters.append(layer.params['b'])
        return parameters

    def save_model(self, path, epoch):
        params = {}
        params_W = self.get_weights()
        params_b = self.get_biases()
        for i in range(len(params_W)):
            params[f'W{i}'] = params_W[i]
            params[f'b{i}'] = params_b[i]
        np.savez(os.path.join(path, f'model_{epoch}.npz'), **params)

    def load_model(self, path, epoch):
        params = np.load(os.path.join(path, f'model_{epoch}.npz'))
        for i, layer in enumerate(self.fc_layers):
            layer.params['W'] = params[f'W{i}']
            layer.params['b'] = params[f'b{i}']
