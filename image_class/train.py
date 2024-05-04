from config import *
from fashion_mnist.utils.mnist_reader import load_mnist
from loss_function import CrossEntropyLoss
from op import FCNet3
from optimizer import SGDOptimizer
from metric import accuracy
from plot_metrics import plot_metrics
import os


def train(model, loss_function, optimizer, train_data, dev_data=None, epochs=10):
    best_score = 0
    train_losses, train_scores, dev_losses, dev_scores = [], [], [], []
    for epoch in range(epochs):
        logits = model(train_data['X'])
        train_loss = loss_function(logits, train_data['y'])
        train_score = accuracy(logits, train_data['y'])
        train_losses.append(train_loss)
        train_scores.append(train_score)
        loss_function.backward()
        optimizer.update(epoch)
        if dev_data is not None:
            logits = model(dev_data['X'])
            dev_loss = loss_function(logits, dev_data['y'])
            dev_score = accuracy(logits, dev_data['y'])
            dev_losses.append(dev_loss)
            dev_scores.append(dev_score)
            if epoch < 10: continue
            if dev_score > best_score + 0.01:
                best_score = dev_score
                if not os.path.exists(SAVE_DIR):
                    os.makedirs(SAVE_DIR)
                model.save_model(SAVE_DIR, epoch)
                print(f"Epoch: {epoch}/{epochs}, Train loss: {train_loss:.4f}, dev score: {dev_score:.4f}")
        if epoch and epoch % 50 == 0:
            plot_metrics(train_scores, train_losses, dev_scores, dev_losses)
    plot_metrics(train_scores, train_losses, dev_scores, dev_losses, save_path=SAVE_DIR)


if __name__ == '__main__':
    # 数据
    X_train, y_train = load_mnist('./fashion_mnist/data/fashion', kind='train')  # (60000, 784), (60000,)
    X_test, y_test = load_mnist('./fashion_mnist/data/fashion', kind='t10k')

    traindata = {'X': X_train / 255., 'y': y_train}
    devdata = {'X': X_test / 255., 'y': y_test}
    # 模型
    fcnet = FCNet3(in_features=IN_FEATURES, hidden_size1=HIDDEN_SIZE_1, hidden_size2=HIDDEN_SIZE_2,
                   out_features=OUT_FEATURES)
    # 损失函数
    loss_fn = CrossEntropyLoss(fcnet)
    # 优化器
    sgd = SGDOptimizer(init_lr=LR, model=fcnet, lambda_l2=L2, decay_rate=DECAY_RATE, decay_steps=DECAY_STEP)
    train(fcnet, loss_fn, sgd, traindata, devdata, epochs=EPOCH)

