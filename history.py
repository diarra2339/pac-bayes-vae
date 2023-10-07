from collections import defaultdict
import matplotlib.pyplot as plt


class History:
    def __init__(self, logs_list=None):
        self.history = defaultdict(list)

        if logs_list is not None:
            for logs in logs_list:
                self.save(logs)

    def save(self, logs):
        for k, v in logs.items():
            self.history[k].append(v)

    def __len__(self):
        return max([len(self.history[key]) for key in self.history.keys()])

    def __getitem__(self, key):
        return self.history[key]

    def show_losses(self, losses=None, start_epoch=1):
        losses = list(self.history.keys()) if losses is None else losses
        num_plots = len(losses)
        _, axes = plt.subplots(nrows=num_plots, ncols=1, sharex=True)
        plt.tight_layout()
        epochs = len(self[losses[0]])
        epochs = list(range(start_epoch, epochs + 1))

        for i in range(num_plots):
            axes[i].set_xlabel('Epochs')
            axes[i].set_ylabel(losses[i])
            axes[i].plot(epochs, self[losses[i]][start_epoch - 1:], label=losses[i])
        plt.show()
