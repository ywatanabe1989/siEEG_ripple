#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import mngs


class MNet_100(nn.Module):
    def __init__(self, config):
        super().__init__()

        # basic
        self.config = config
        self.n_chs = len(self.config["montage"])

        # conv
        self.conv1 = nn.Conv2d(1, 40, kernel_size=(6, 4))
        self.act1 = nn.Mish()        
        
        self.conv2 = nn.Conv2d(40, 40, kernel_size=(1, 4))
        self.bn2 = nn.BatchNorm2d(40)
        self.pool2 = nn.MaxPool2d((1, 5))
        self.act2 = nn.Mish()                

        self.swap = SwapLayer()
        
        self.conv3 = nn.Conv2d(1, 50, kernel_size=(8, 12))
        self.bn3 = nn.BatchNorm2d(50)
        self.pool3 = nn.MaxPool2d((3, 3))
        self.act3 = nn.Mish()        
        
        self.conv4 = nn.Conv2d(50, 50, kernel_size=(3, 1))
        self.bn4 = nn.BatchNorm2d(50)
        self.pool4 = nn.MaxPool2d((1, 2))
        self.act4 = nn.Mish()                


        # fc
        n_fc_in = 450 # 15950
        
        self.fc_diag = nn.Sequential(
            nn.Linear(n_fc_in, config["n_fc1"]),
            nn.Mish(),
            # nn.BatchNorm1d(config["n_fc1"]),
            nn.Dropout(config["d_ratio1"]),
            nn.Linear(config["n_fc1"], config["n_fc2"]),
            nn.Mish(),
            # nn.BatchNorm1d(config["n_fc2"]),
            nn.Dropout(config["d_ratio2"]),
            nn.Linear(config["n_fc2"], len(config["class_labels"])),
        )

        self.fc_subj = nn.Sequential(
            nn.Linear(n_fc_in, config["n_fc1"]),
            nn.Mish(),
            # nn.BatchNorm1d(config["n_fc1"]),
            nn.Dropout(config["d_ratio1"]),
            nn.Linear(config["n_fc1"], config["n_fc2"]),
            nn.Mish(),
            # nn.BatchNorm1d(config["n_fc2"]),
            nn.Dropout(config["d_ratio2"]),
            nn.Linear(config["n_fc2"], config["n_subjs_tra"]),
        )
        
    @staticmethod
    def _reshape_input(x, n_chs):
        # (batch, channel, time_length) ->
        # (batch, channel, time_length, new_axis)
        if x.ndim == 3:
            x = x.unsqueeze(-1)
        if x.shape[2] == n_chs:
            x = x.transpose(1, 2)
        x = x.transpose(1, 3).transpose(2, 3)
        return x

    @staticmethod
    def _normalize_time(x):
        return (x - x.mean(dim=-1, keepdims=True)) \
            / x.std(dim=-1, keepdims=True)

    def forward(self, x):
        # bs = x.shape[0]
        # x.shape # [16, 19, 100]
        
        # time-wise normalization
        x = self._normalize_time(x)

        x = self._reshape_input(x, self.n_chs)

        x = self.act1(self.conv1(x))
        x = self.act2(self.pool2(self.bn2(self.conv2(x))))
        x = self.swap(x)
        x = self.act3(self.pool3(self.bn3(self.conv3(x))))
        x = self.act4(self.pool4(self.bn4(self.conv4(x))))
        x = x.reshape(len(x), -1)

        y_diag = self.fc_diag(x)
        y_subj = self.fc_subj(x)        
        
        return y_diag, y_subj

class SwapLayer(nn.Module):
    def __init__(self, ):
        super().__init__()
    def forward(self, x):
        return x.transpose(1, 2)        



if __name__ == "__main__":
    ## Demo data
    bs, n_chs, seq_len = 16, 6, 100
    inp = torch.rand(bs, n_chs, seq_len)
    Ab = torch.rand(bs, )
    Sb = torch.rand(bs, )
    Mb = torch.rand(bs, )    

    ## Config for the model
    model_config = mngs.general.load("eeg_human_ripple_clf/models/MNet/MNet_100.yaml")
    MONTAGE_8 = mngs.general.load("./config/global.yaml")["EEG_8_COMMON_CHANNELS"]
    MONTAGE_6 = MONTAGE_8[:6]
    
    model_config.update(
        dict(
            class_labels=["NoN-Ripple", "Ripple"],
            montage=MONTAGE_6,
            n_subjs_tra=1024,
        )
    )
    model = MNet_100(model_config).cuda()

    y_diag, y_subj = model(inp.cuda())
    
    summary(model, inp)
    # print(y.shape)
