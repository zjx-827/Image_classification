import matplotlib.pyplot as plt


def plot_metrics(train_scores, train_losses, dev_scores, dev_losses, save_path=None):
    epochs = range(1, len(train_scores) + 1)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_scores, 'b', label='Train Score')
    plt.plot(epochs, dev_scores, 'r', label='Dev Score')
    plt.title('Training and Dev Scores per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses, 'b', label='Train Loss')
    plt.plot(epochs, dev_losses, 'r', label='Dev Loss')
    plt.title('Training and Dev Losses per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
