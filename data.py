import torch
import torch.distributions as D
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np

from utils import heat_scatter


class ToyData2D:
    def __init__(self):
        pass

    def plot_data(self, num_samples, style='scatter', sigma=32, name='Toy Data', r=4):
        num_samples = min(num_samples, len(self.train_data))
        data = self.train_data.tensors[0][: num_samples]
        x, y = data[:, 0], data[:, 1]
        if style.startswith('heat'):
            heat_scatter(x=x, y=y, sigma=sigma, r=r)
        else:
            plt.scatter(x=x, y=y, alpha=0.5)
        plt.show()



class GMM:
    def __init__(self, coeffs, means, variances):
        # means and variances have the same form: list of lists of floats or list of Tensors
        if type(means) is list and type(means[0]) is list:
            means = [torch.Tensor(x) for x in means]
            variances = [torch.Tensor(x) for x in variances]
            means = torch.stack(means)
            variances = torch.stack(variances)
            self.dimension = len(means[0])
        elif type(means) is list and type(means[0]) is torch.Tensor:
            means = torch.stack(means)
            variances = torch.stack(variances)
        elif type(means) is torch.Tensor:
            pass  # use means and variances as is
        else:
            raise ValueError('The variables means and variances do not have the right type')

        mix = D.Categorical(torch.Tensor(coeffs))
        comp = D.Independent(D.Normal(means, torch.sqrt(variances)), 1)
        self.means = means
        self.variances = variances
        self.gmm = D.MixtureSameFamily(mix, comp)

    def sample(self, num_samples):
        return self.gmm.sample([num_samples])


class TwoGaussians(ToyData2D):
    def __init__(self, centers_x=(-1, 1), sigma=0.05, train_size=40000, val_size=20000, test_size=20000, batch_size=64, workers=7):
        super(TwoGaussians, self).__init__()
        self.width, self.height = centers_x[1] - centers_x[0], 0
        self.max_width, self.max_height = self.width + 6*sigma, self.height + 6*sigma
        self.centers_x = centers_x if centers_x is not None else (-self.width / 4, self.width / 4)
        self.sigma = sigma
        self.diameter = np.sqrt(self.max_width**2 + self.max_height**2)

        width_margin = (self.max_width - self.width) / 2
        height_margin = (self.max_height - self.height) / 2
        self.width_min_, self.width_max_ = -self.width / 2 - width_margin, self.width / 2 + width_margin
        self.height_min_, self.height_max_ = -self.height / 2 - height_margin, self.height / 2 + height_margin

        self.set_loaders(train_size=train_size, val_size=val_size, test_size=test_size, batch_size=batch_size, workers=workers)

    def truncate(self, data):
        data = data[torch.vstack([(self.width_min_ <= data[:, 0]), (data[:, 0] <= self.width_max_),
                                  (self.height_min_ <= data[:, 1]), (data[:, 1] <= self.height_max_)]).all(axis=0)]
        return data

    def set_loaders(self, train_size, val_size, test_size, batch_size, workers):
        gmm = GMM(coeffs=[1, 1], means=[[self.centers_x[0], 0], [self.centers_x[1], 0]], variances=[[self.sigma ** 2, self.sigma ** 2]] * 2)
        data = gmm.sample(train_size + val_size + test_size + int(0.5 * train_size))
        data = self.truncate(data)
        assert data.shape[0] >= train_size + val_size + test_size

        self.train_data = TensorDataset(data[:train_size], torch.zeros(train_size, 1))
        self.val_data = TensorDataset(data[train_size: train_size + val_size], torch.zeros(val_size, 1))
        self.test_data = TensorDataset(data[train_size + val_size: train_size + val_size + test_size], torch.zeros(test_size, 1))
        self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True, num_workers=workers)
        self.val_loader = DataLoader(self.val_data, batch_size=batch_size, shuffle=True, num_workers=workers)
        self.test_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=True, num_workers=workers)

    def plot_data(self, num_samples, style='scatter', sigma=32, name='Toy Data', plot_rect=False, r=4):
        super(TwoGaussians, self).plot_data(num_samples=num_samples, style=style, sigma=sigma, name=name, r=r)
        if plot_rect:
            square = patches.Rectangle(xy=(-self.max_width/2, -self.max_height/2), width=self.max_width, height=self.max_height,
                                       fill=False, edgecolor='green', linewidth=2)
            plt.gca().add_patch(square)
        plt.show()


class Circle(ToyData2D):
    def __init__(self, radius=3, max_radius=3.2, sigma=0.05,
                 train_size=60000, val_size=30000, test_size=30000, batch_size=256, workers=5):
        super(Circle, self).__init__()
        self.radius = radius
        self.max_radius = max_radius
        self.diameter = 2 * self.max_radius
        self.sigma = sigma

        self.set_loaders(train_size, val_size, test_size, batch_size, workers)

    def set_loaders(self, train_size, val_size, test_size, batch_size, workers):
        n = train_size + val_size + test_size
        n += n//2
        linspace = torch.linspace(start=0, end=2 * np.pi, steps=n)
        all_data = torch.stack([self.radius * torch.cos(linspace), self.radius * torch.sin(linspace)], dim=1)
        all_data += self.sigma * torch.randn_like(all_data)
        all_data = self.truncate(all_data)
        assert all_data.shape[0] >= train_size + val_size + test_size
        all_data = all_data[torch.randperm(all_data.shape[0])]

        self.train_data = TensorDataset(all_data[:train_size], torch.zeros(train_size, 1))
        self.val_data = TensorDataset(all_data[train_size: train_size + val_size], torch.zeros(val_size))
        self.test_data = TensorDataset(all_data[train_size + val_size: train_size + val_size + test_size], torch.zeros(test_size))
        self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True, num_workers=workers)
        self.val_loader = DataLoader(self.val_data, batch_size=batch_size, shuffle=True, num_workers=workers)
        self.test_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=True, num_workers=workers)

    def truncate(self, data):
        data = data[data[:, 0]**2 + data[:, 1]**2 <= self.max_radius**2]
        return data

    def plot_data(self, num_samples, style='scatter', sigma=32, name='Toy Data', circle=False, r=4):
        super(Circle, self).plot_data(num_samples=num_samples, style=style, sigma=sigma, name=name, r=r)
        if circle:
            ax = plt.gca()
            c = plt.Circle(xy=(0, 0), radius=self.max_radius, color='r', fill=False)
            ax.add_patch(c)
        plt.show()

