import numpy as np


class LossFunction:
    def __init__(self):
        self.preds = None
        self.labels = None

    def __call__(self, preds, labels):
        return self.forward(preds, labels)

    def forward(self, preds, labels):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class CrossEntropyLoss(LossFunction):
    def __init__(self, model):
        super(CrossEntropyLoss, self).__init__()
        self.model = model
        self.N = None

    def forward(self, preds, labels):
        """
        :param preds: shape = [batch_size, out_features]
        :param labels: shape = [batch_size]
        :return:
        """
        self.preds = softmax(preds)
        self.labels = labels
        self.N = labels.shape[0]
        loss = 0
        for i in range(self.N):
            index = self.labels[i]
            loss -= np.log(self.preds[i, index] + 1e-8)
        loss /= self.N
        return loss

    def backward(self):
        grad = np.zeros_like(self.preds)
        for i in range(self.N):
            index = self.labels[i]
            grad[i, :] = self.preds[i, :]
            grad[i, index] -= 1
        grad /= self.N
        self.model.backward(grad)
        # return grad
