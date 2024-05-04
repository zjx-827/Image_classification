import numpy as np

np.random.seed(1005)


def accuracy(preds, labels):
    preds = np.argmax(preds, axis=1)
    return np.mean(np.equal(preds, labels))
