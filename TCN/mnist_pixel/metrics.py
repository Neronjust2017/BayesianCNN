import numpy as np
import torch.nn.functional as F
from torch import nn
import torch


class ELBO(nn.Module):
    def __init__(self, train_size):
        super(ELBO, self).__init__()
        self.train_size = train_size

    def forward(self, input, target, kl, kl_weight=0.1):
        # loss = criterion(log_outputs, labels, kl)
        assert not target.requires_grad
        # return F.nll_loss(input, target, size_average=True) * self.train_size + kl_weight * kl
        kl_loss = kl_weight * kl
        likihood_loss = F.nll_loss(input, target, size_average=True) * self.train_size
        return likihood_loss, kl_loss

class ELBO_regression(nn.Module):
    def __init__(self, train_size):
        super(ELBO_regression, self).__init__()
        self.train_size = train_size

    def forward(self, input, target, sigma, no_dim, kl, kl_weight=0.1):
        # loss = criterion(log_outputs, labels, kl)
        assert not target.requires_grad
        return  log_gaussian_loss(input, target,sigma, no_dim)* self.train_size + kl_weight * kl


def lr_linear(epoch_num, decay_start, total_epochs, start_value):
    if epoch_num < decay_start:
        return start_value
    return start_value*float(total_epochs-epoch_num)/float(total_epochs-decay_start)


def acc(outputs, targets):
    return np.mean(outputs.cpu().numpy().argmax(axis=1) == targets.data.cpu().numpy())

def mse(outputs, targets):
    return F.mse_loss(outputs, targets, size_average=True)

def calculate_kl(log_alpha):
    return 0.5 * torch.sum(torch.log1p(torch.exp(-log_alpha)))


def log_gaussian_loss(output, target, sigma, no_dim):
    exponent = -0.5 * (target - output) ** 2 / sigma ** 2
    log_coeff = -no_dim * torch.log(sigma)

    return - (log_coeff + exponent).sum()

