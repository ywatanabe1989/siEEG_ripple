#!/usr/bin/env python
# https://github.com/mravanelli/SincNet/blob/master/dnn_models.py
import torch
import torch.nn as nn
import numpy as np

class LayerNorm(nn.Module):

    def __init__(self, input_shape, dim=-1, eps=1e-6):
        super(LayerNorm,self).__init__()
        self.gamma = nn.Parameter(torch.ones(input_shape))
        self.beta = nn.Parameter(torch.zeros(input_shape))
        self.register_buffer('eps', torch.tensor(np.array(eps)))
        self.dim = dim

    def forward(self, x):
        mean = x.mean(dim=self.dim, keepdim=True)
        std = x.std(dim=self.dim, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta    
