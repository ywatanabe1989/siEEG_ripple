#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNet_1000(nn.Module):
    def __init__(self, config):
        super(MNet_1000, self).__init__()
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

        ## diagnosis predict fc
        self.fc_diag = nn.Sequential(
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

        ## subject predict fc
        self.fc_subj = nn.Sequential(
            nn.Linear(15950, config["n_fc1"]),
            nn.ReLU(),
            # nn.BatchNorm1d(config["n_fc1"]),
            nn.Dropout(config["d_ratio1"]),
            nn.Linear(config["n_fc1"], config["n_fc2"]),
            nn.ReLU(),
            # nn.BatchNorm1d(config["n_fc2"]),
            nn.Dropout(config["d_ratio2"]),
            nn.Linear(config["n_fc2"], config["n_subj_tra_uq"]),
        )

    @staticmethod
    def _reshape_input(x, n_chs=19):
        # # (batch, channel, time_length) ->  (batch, channel, time_length, new_axis)
        if x.ndim == 3:
            x = x.unsqueeze(-1)
        if x.shape[2] == n_chs:
            x = x.transpose(1, 2)
        x = x.transpose(1, 3).transpose(2, 3)
        return x

    def forward(self, x):
        bs = x.shape[0]
        x = self._reshape_input(x)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.bn1(x)
        x = self.pool1(x)
        x = x.transpose(1, 2)
        x = F.relu(self.conv3(x))
        x = self.bn2(x)
        x = self.pool2(x)
        x = F.relu(self.conv4(x))
        x = self.bn3(x)
        x = self.pool3(x)
        x = x.transpose(1, 3)
        x = x.reshape(bs, -1)

        y1 = self.fc_diag(x)

        y2 = self.fc_subj(x)

        return y1, y2


if __name__ == "__main__":
    import mngs

    ## Demo data
    bs, n_chs, seq_len = 16, 19, 1000
    inp = torch.rand(bs, n_chs, seq_len)

    ## Config for the model
    model_config = mngs.general.load("eeg_dem_clf/models/MNet/MNet_1000.yaml")
    disease_types = ["HV", "AD", "DLB", "NPH"]
    n_subj_tra_uq = 1024
    model_config.update(
        dict(
            disease_types=disease_types,
            n_subj_tra_uq=n_subj_tra_uq,
        )
    )
    mnet = MNet_1000(model_config)
    y1, y2 = mnet(inp)
    print(y1.shape)
