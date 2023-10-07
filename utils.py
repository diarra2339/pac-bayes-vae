import torch

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

from omegaconf import OmegaConf
import json


def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_config(filename='config.json'):
    with open(filename, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    main_conf = OmegaConf.create(json_data)
    main_conf.cuda = torch.cuda.is_available()
    return main_conf


def heat_scatter(x, y, sigma=16, r=4, ax=None, symzero=True):
    bins = x.shape[0] // 2
    range_ = [[-r, r], [-r, r]] if symzero else [[-1, r], [-1, r]]
    heatmap, xedges, yedges = np.histogram2d(x.numpy(), y.numpy(), bins=bins, range=range_)
    heatmap = gaussian_filter(heatmap, sigma=sigma)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    if ax is None:
        plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='BuGn')
    else:
        ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='BuGn')


