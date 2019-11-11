import torch.nn as nn
import torch
import torch.nn.functional as F


class Mish(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
