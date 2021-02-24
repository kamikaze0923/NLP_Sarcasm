import os
import matplotlib.pyplot as plt


class Training_Info_Buffer:

    def __init__(self):
        self.train_loss_buffer = []
        self.validate_loss_buffer = []
        self.train_acc_buffer = []
        self.validate_acc_buffer = []

    def add_info(self, loss, acc):
        train_loss, validate_loss = loss
        train_acc, validate_acc = acc
        self.train_loss_buffer.append(train_loss)
        self.validate_loss_buffer.append(validate_loss)
        self.train_acc_buffer.append(train_acc)
        self.validate_acc_buffer.append(validate_acc)


def plot_loss(working_dir, buffer):
    loss_begin, acc_begin = buffer.validate_loss_buffer[0], buffer.validate_loss_buffer[0]

    plt.plot(buffer.train_loss_buffer, c='b')
    plt.plot(buffer.validate_loss_buffer, c='r')
    plt.scatter(0, loss_begin, marker="^", c='k')
    best_valid_loss = min(buffer.validate_loss_buffer)
    scatter_x, scatter_y = buffer.validate_loss_buffer.index(best_valid_loss), best_valid_loss
    plt.scatter(scatter_x, scatter_y, marker="^", c='k')
    plt.annotate(text=f"{best_valid_loss: .2f}", xy=(scatter_x, scatter_y), c='k')
    plt.grid()
    plt.legend(["Train_loss", "Valid_loss", "Valid_loss_begin", "Valid_loss_best"])
    plt.savefig(os.path.join(working_dir, "loss.png"))
    plt.close()


    plt.plot(buffer.train_acc_buffer)
    plt.plot(buffer.validate_acc_buffer)
    plt.scatter(0, acc_begin, marker="^", c='k')
    best_valid_acc = max(buffer.validate_acc_buffer)
    scatter_x, scatter_y = buffer.validate_acc_buffer.index(best_valid_acc), best_valid_acc
    plt.scatter(scatter_x, scatter_y, marker="^", c='k')
    plt.annotate(text=f"{best_valid_acc: .2f}", xy=(scatter_x, scatter_y), c='k')
    plt.grid()
    plt.legend(["Train_acc", "Valid_acc", "Valid_acc_begin", "Valid_acc_best"])
    plt.savefig(os.path.join(working_dir, "accuracy.png"))
    plt.close()