class SGDOptimizer:
    def __init__(self, init_lr, model,lambda_l2=0.01, decay_rate=0.9, decay_steps=100):
        self.lr = init_lr
        self.model = model
        self.lambda_l2 = lambda_l2
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

    def update(self, epoch):
        if epoch and epoch % self.decay_steps == 0:
            print(f'Learning rate: {self.lr} --> {self.lr * self.decay_rate}')
            self.lr *= self.decay_rate

        for layer in self.model.fc_layers:
            for key in layer.params:
                layer.params[key] -= self.lr * (layer.grads[key] + self.lambda_l2 * layer.params[key])
