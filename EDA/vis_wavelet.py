#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-10-18 11:32:46 (ywatanabe)"

import sys
sys.path.append("./externals/ripple_detection/")
from ripple_detection.detectors import Kay_ripple_detector
from ripple_detection.core import filter_ripple_band
import numpy as np
import mngs
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def plot_scalogram(i_sub_str, i_session_str, low_hz, high_hz, mean_or_std_str):
    ldir = f"./data/Sub_{i_sub_str}/Session_{i_session_str}/"

    cwts = mngs.io.load(ldir + "cwts.npy")
    freqs = mngs.io.load(ldir + "freqs.npy")    
    
    # z-norm per 8-sec segment
    # cwts = abs(cwts) / abs(np.array(cwts)).max(axis=-1, keepdims=True)
    _cwts_mean = cwts.mean(axis=-1, keepdims=True)
    _cwts_std = cwts.std(axis=-1, keepdims=True)
    cwts_z = (cwts - _cwts_mean) / _cwts_std
    
    time = np.arange(cwts.shape[-1]) / SAMP_RATE_iEEG

    cwt_mean_df = pd.DataFrame(
        data=cwts_z.reshape(-1, cwts_z.shape[-2], cwts_z.shape[-1]).mean(axis=0),
        index=np.array(freqs).round(3),
        columns=time,
        )

    cwt_std_df = pd.DataFrame(
        data=cwts_z.reshape(-1, cwts_z.shape[-2], cwts_z.shape[-1]).std(axis=0),
        index=np.array(freqs).round(3),
        columns=time,
        )

    # sort y axis (freq)
    cwt_mean_df = cwt_mean_df.loc[cwt_mean_df.index[::-1]]
    cwt_std_df = cwt_std_df.loc[cwt_std_df.index[::-1]]    

    # narrow down freqs to show
    # low_hz = 80
    # high_hz = 140
    indi = (low_hz < cwt_mean_df.index) & (cwt_mean_df.index < high_hz)
    
    fig, ax = plt.subplots()
    data = cwt_mean_df[indi] if mean_or_std_str == "mean" else cwt_std_df[indi]
    edge_sec = 0.1
    edge_pts = int(edge_sec*SAMP_RATE_iEEG)
    data = data.iloc[:, edge_pts:-edge_pts]

    sns.heatmap(
        data,
        # vmin=data.min().min(),
        # vmax=data.max().max(),
        ax=ax,
        )
    title = f"Subject_{i_sub_str}_Session_{i_session_str}"
    ax.set_title(f"Subject: {i_sub_str}\nSession: {i_session_str}")
    # plt.show()
    mngs.io.save(fig, f"./tmp/figs_wavelet/{low_hz}-{high_hz}Hz/{mean_or_std_str}/{title}.png")
    plt.close()

if __name__ == "__main__":
    from glob import glob
    from natsort import natsorted
    import re
    import seaborn as sns    

    SAMP_RATE_iEEG = 2000
    
    sub_dirs = natsorted(glob("./data/Sub_*"))
    indi_subs_str = [re.match("./data/Sub_\w{2}", sd).string[-2:] for sd in sub_dirs]

    # i_sub_str = "02"
    # i_session_str = "01"

    for i_sub_str in indi_subs_str:
        session_dirs = natsorted(glob(f"./data/Sub_{i_sub_str}/Session_*"))
        indi_sessions_str = [re.match("./data/Sub_\w{2}/Session_\w{2}", sd).string[-2:] for sd in session_dirs]
        for i_session_str in indi_sessions_str:
            for (low_hz, high_hz) in [(80, 140), (1, 500)]:
                for mean_or_std_str in ["mean", "std"]:
                    plot_scalogram(i_sub_str, i_session_str, low_hz, high_hz, mean_or_std_str)
