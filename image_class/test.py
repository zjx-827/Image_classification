from config import *
from fashion_mnist.utils.mnist_reader import load_mnist
from op import FCNet3
from loss_function import CrossEntropyLoss
from metric import accuracy

if __name__ == '__main__':
    X_test, y_test = load_mnist('./fashion_mnist/data/fashion', kind='t10k')
    devdata = {'X': X_test / 255., 'y': y_test}
    fcnet = FCNet3(in_features=IN_FEATURES, hidden_size1=HIDDEN_SIZE_1, hidden_size2=HIDDEN_SIZE_2,
                   out_features=OUT_FEATURES)
    fcnet.load_model(SAVE_DIR, TEST_EPOCH)
    loss_fn = CrossEntropyLoss(fcnet)
    logits = fcnet(devdata['X'])
    dev_loss = loss_fn(logits, devdata['y'])
    dev_score = accuracy(logits, devdata['y'])
    print(dev_loss, dev_score)
