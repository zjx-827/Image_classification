import numpy as np


class Activation(object):
    def __init__(self):
        self.inputs = None

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, inputs):
        raise NotImplementedError


class ReLU(Activation):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, inputs):
        """
        :param inputs: shape = [batch_size, input_size]
        :return: shape = [batch_size, input_size]
        """
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, grads_out):
        """
        :param grads_out: shape = [batch_size, output_size]
        :return: grads_in: shape = [batch_size, output_size]
        """
        grads_in = grads_out.copy()
        grads_in[self.inputs < 0] = 0
        return grads_in


class Logistic(Activation):
    def __init__(self):
        super(Logistic, self).__init__()

    def forward(self, inputs):
        """
        :param inputs: shape = [batch_size, input_size]
        :return: shape = [batch_size, input_size]
        """
        self.inputs = inputs
        return 1 / (1 + np.exp(-inputs))

    def backward(self, grads_out):
        """
        :param grads_out: shape = [batch_size, output_size]
        :return: grads_in: shape = [batch_size, output_size]
        """
        sigmoid = 1 / (1 + np.exp(-self.inputs))
        grads_in = grads_out * (sigmoid * (1 - sigmoid))
        return grads_in
