#!/usr/bin/env python

import sys

import torch
import torch.nn as nn
from eeg_dem_clf.models.modules.act_funcs.init_act_layer import init_act_layer
from eeg_dem_clf.models.modules.LayerNorm import LayerNorm


class BasicBlock(nn.Module):
    # conv2d -> BN -> maxpool2d
    def __init__(
        self,
        in_chs,
        out_chs,
        conv_ks,
        act_str,
        n_last_of_input,
        mp_ks,
    ):

        super(BasicBlock, self).__init__()

        # [bs, RGB_CH, H, W]
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size=conv_ks)
        self.act = init_act_layer(act_str)  # nn.ReLU()
        self.ln = LayerNorm(n_last_of_input - (conv_ks[1] - 1))
        # self.bn = nn.BatchNorm2d(out_chs)
        self.max_pooling = nn.MaxPool2d(mp_ks)  # max_pooling_kernel_size

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.ln(x)
        # x = self.bn(x)
        x = self.max_pooling(x)
        return x
