import numpy as np
import torch

from loss import BoundLoss
from utils import get_device


# Thm 4.3
def recons_bound_diam(model, val_loader, test_loader, lamda, k_phi, k_theta, delta, diameter):
    # val_loader is the validation data-loader, used to compute the rhs of the bound.
    # test_loader is used to estimate the expected reconstruction loss
    n = len(val_loader.dataset)
    loss_fct = BoundLoss()

    # Compute the rhs
    rec_loss, kl_div = torch.Tensor([0]).to(get_device()), torch.Tensor([0]).to(get_device())
    for x, _ in val_loader:
        x = x.to(get_device())
        with torch.no_grad():
            z, mu, logvar = model.encode(x)
            x_hat = model.decode(z)
        rec_loss += loss_fct.recons_loss_sum(x=x, x_hat=x_hat)
        kl_div += loss_fct.kl_divergence_sum(mu=mu, logvar=logvar)

    rec_loss = rec_loss / n
    kl_div = kl_div / lamda
    avg_distance = (k_phi * k_theta * diameter)
    exp_moment = (np.log(1/delta) / lamda) + (lamda * diameter**2 / (8*n))
    bound = rec_loss + kl_div + avg_distance + exp_moment
    dico = {'rec_loss': rec_loss, 'kl_div': kl_div, 'avg_distance': avg_distance, 'exp_moment': exp_moment, 'bound': bound}

    # Estimate the lhs with the test set
    test_size = len(test_loader.dataset)
    rec_loss_test = torch.Tensor([0]).to(get_device())
    for x, _ in test_loader:
        x = x.to(get_device())
        with torch.no_grad():
            z, mu, logvar = model.encode(x)
            x_hat = model.decode(z)
        rec_loss_test += loss_fct.recons_loss_sum(x=x, x_hat=x_hat)
    rec_loss_test = rec_loss_test / test_size

    dico['rec_loss_test'] = rec_loss_test

    return dico
