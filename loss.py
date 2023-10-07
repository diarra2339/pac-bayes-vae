import torch
import torch.nn.functional as F


class Loss:
    def __init__(self, criterion):
        assert criterion in ['mse', 'rmse', 'bce']
        self.criterion = criterion

    @staticmethod
    def kl_divergence(mu, logvar):
        return torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)

    def recons_loss(self, x, x_hat):
        batch_size = x.shape[0]
        x = x.view(x_hat.shape)
        if self.criterion == 'mse':
            rec_loss = F.mse_loss(target=x, input=x_hat, reduction='sum') / batch_size
        elif self.criterion == 'rmse':
            rec_loss = torch.sum(torch.norm(x - x_hat, p=2, dim=1)) / batch_size
        elif self.criterion == 'bce':
            rec_loss = F.binary_cross_entropy(target=x, input=x_hat)
        return rec_loss

    def __call__(self, x, x_hat, mu, logvar):
        rec_loss = self.recons_loss(x=x, x_hat=x_hat)
        kl_div = self.kl_divergence(mu=mu, logvar=logvar)
        return rec_loss, kl_div


# Loss used for computing the PAC-Bayes bounds; uses the L2 norm
class BoundLoss:
    def __init__(self):
        self.criterion = 'rmse'

    @staticmethod
    def recons_loss_sum(x, x_hat):
        # returns the sum of the reconstruction losses, used for computing the bound in batches
        # Assumes x and x_hat are 2-dimensional, so linear data.
        rec_loss = torch.sum(torch.norm(x - x_hat, p=2, dim=1))
        return rec_loss

    @staticmethod
    def kl_divergence_sum(mu, logvar):
        return torch.sum(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1))

