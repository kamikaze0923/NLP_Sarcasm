import os
import matplotlib.pyplot as plt


class Training_Info_Buffer:

    def __init__(self):
        self.train_loss_buffer = []
        self.validate_loss_buffer = []
        self.train_acc_buffer = []
        self.validate_acc_buffer = []


def plot_loss(working_dir, buffer, test_result, best_acc_result, args):
    test_loss, test_acc = test_result

    plt.plot(buffer.train_loss_buffer)
    plt.plot(buffer.validate_loss_buffer)
    plt.scatter(args.epochs - 1, test_loss, marker="^", c='k')
    plt.legend(["Train_loss", "Valid_loss", "Test_loss"])
    plt.savefig(os.path.join(working_dir, "loss.png"))
    plt.close()

    best_acc, best_epoch = best_acc_result
    plt.title(f"Best valid acc: {best_acc:.2f} in epoch {best_epoch}")
    plt.plot(buffer.train_acc_buffer)
    plt.plot(buffer.validate_acc_buffer)
    plt.scatter(args.epochs - 1, test_acc, marker="^", c='k')
    plt.legend(["Train_acc", "Valid_acc", "Test_acc"])
    plt.savefig(os.path.join(working_dir, "accuracy.png"))
    plt.close()