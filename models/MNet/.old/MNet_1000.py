#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import mngs


class MNet_1000(nn.Module):
    def __init__(self, config):
        super(MNet_1000, self).__init__()

        self.config = config
        self.n_chs = len(self.config["montage"])

        self.input_bn = nn.BatchNorm2d(19)

        self.act = mngs.ml.act.define(config["act_str"])

        self.conv1 = nn.Conv2d(1, 40, kernel_size=(19, 4))
        self.conv2 = nn.Conv2d(40, 40, (1, 4))
        self.bn1 = nn.BatchNorm2d(40)
        self.pool1 = nn.MaxPool2d((1, 5))
        self.conv3 = nn.Conv2d(1, 50, (8, 12))
        self.bn2 = nn.BatchNorm2d(50)
        self.pool2 = nn.MaxPool2d((3, 3))
        self.conv4 = nn.Conv2d(50, 50, (1, 5))
        self.bn3 = nn.BatchNorm2d(50)
        self.pool3 = nn.MaxPool2d((1, 2))

        ## fc
        self.fc = nn.Sequential(
            nn.Linear(15950, config["n_fc1"]),
            nn.ReLU(),
            # nn.BatchNorm1d(config["n_fc1"]),
            nn.Dropout(config["d_ratio1"]),
            nn.Linear(config["n_fc1"], config["n_fc2"]),
            nn.ReLU(),
            # nn.BatchNorm1d(config["n_fc2"]),
            nn.Dropout(config["d_ratio2"]),
            nn.Linear(config["n_fc2"], len(config["disease_types"])),
        )

    @staticmethod
    def _reshape_input(x, n_chs):
        # (batch, channel, time_length) ->  (batch, channel, time_length, new_axis)
        if x.ndim == 3:
            x = x.unsqueeze(-1)
        if x.shape[2] == n_chs:
            x = x.transpose(1, 2)
        x = x.transpose(1, 3).transpose(2, 3)
        return x

    @staticmethod
    def _normalize_time(x):
        return (x - x.mean(dim=-1, keepdims=True)) / x.std(dim=-1, keepdims=True) # NaN
        # return (x - x.mean(dim=-1, keepdims=True)) # 0.39
        # return x

    def forward(self, x):
        bs = x.shape[0]

        # x.shape # [16, 19, 1000]
        x = self.input_bn(x.unsqueeze(-1)).squeeze()

        # x = self._normalize_time(x)

        # x = x.clip(min=-2, max=2)

        x = self._reshape_input(x, self.n_chs)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.bn1(x)
        x = self.pool1(x)
        x = x.transpose(1, 2)
        x = self.act(self.conv3(x))
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.act(self.conv4(x))
        x = self.bn3(x)
        x = self.pool3(x)
        x = x.transpose(1, 3)
        x = x.reshape(bs, -1)

        x = self.fc(x)

        return x


if __name__ == "__main__":
    ## Demo data
    bs, n_chs, seq_len = 16, 19, 1000
    inp = torch.rand(bs, n_chs, seq_len)

    ## Config for the model
    model_config = mngs.general.load("eeg_dem_clf/models/MNet/MNet_1000.yaml")
    MONTAGE_19 = [
        "FP1",
        "FP2",
        "F3",
        "F4",
        "C3",
        "C4",
        "P3",
        "P4",
        "O1",
        "O2",
        "F7",
        "F8",
        "T7",
        "T8",
        "P7",
        "P8",
        "Fz",
        "Cz",
        "Pz",
    ]
    model_config.update(
        dict(
            disease_types=["HV", "AD", "DLB", "NPH"],
            montage=MONTAGE_19,
        )
    )
    model = MNet_1000(model_config)
    y = model(inp)
    summary(model, inp)
    print(y.shape)
