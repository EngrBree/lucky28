import torch.distributions as dist
import torch

import torch.nn as nn


class BayesianLoss(nn.Module):
    def __init__(self):
        super(BayesianLoss, self).__init__()

    def forward(self, outputs, targets):
        """Computes a Bayesian cross-entropy loss that considers uncertainty."""
        probs = torch.sigmoid(outputs)
        dist_pred = dist.Bernoulli(probs)
        loss = -dist_pred.log_prob(targets).mean()
        return loss
