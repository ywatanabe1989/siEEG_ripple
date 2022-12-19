import torch.nn as nn

class _None(nn.Module):  # _None activation
    def forward(self, x):
        return x
