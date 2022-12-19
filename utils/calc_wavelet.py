#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-10-25 23:29:17 (ywatanabe)"

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

def load_iEEG(i_sub_str, i_session_str):
    global iEEG, iEEG_ripple_band_passed, SAMP_RATE_iEEG
    trials_info = mngs.io.load(f"./data/Sub_{i_sub_str}/Session_{i_session_str}/trials_info.csv")
    trials_info["set_size"]
    trials_info["correct"] = trials_info["correct"].replace({0:False, 1:True})
    trials_info["match"] = trials_info["match"].replace({1:"match IN", 2:"mismatch OUT"})    
    
    iEEG = mngs.io.load(f"./data/Sub_{i_sub_str}/Session_{i_session_str}/iEEG.pkl")
    SAMP_RATE_iEEG = 2000

    return iEEG
    
if __name__ == "__main__":
    from glob import glob
    from natsort import natsorted
    import re
    import sys
    sys.path.append("./externals/PyTorchWavelets/")
    from wavelets_pytorch.transform import WaveletTransform, WaveletTransformTorch
    from examples.plot import plot_scalogram
    import seaborn as sns
    
    sub_dirs = natsorted(glob("./data/Sub_*"))
    indi_subs_str = [re.match("./data/Sub_\w{2}", sd).string[-2:] for sd in sub_dirs]
    for i_sub_str in indi_subs_str:
        # i_sub_str = "01"

        session_dirs = natsorted(glob(f"./data/Sub_{i_sub_str}/*"))
        indi_sessions_str = [re.match("./data/Sub_\w{2}/Session_\w{2}", sd).string[-2:] for sd in session_dirs]
        for i_session_str in indi_sessions_str:
            # i_session_str = "01"

            iEEG = load_iEEG(i_sub_str=i_sub_str, i_session_str=i_session_str) # (50, 16, 16000)

            # calclate wavelet per session
            cwts = []
            nyq = int(SAMP_RATE_iEEG / 2)
            for i_trial in tqdm(range(iEEG.shape[0])):
                cwts_tmp = []        
                for i_ch in range(iEEG.shape[1]):
                    cwt_mngs = mngs.dsp.wavelet(iEEG[i_trial, i_ch], SAMP_RATE_iEEG, f_min=0.1)
                    cwt_mngs = cwt_mngs[cwt_mngs.index < nyq]
                    cwts_tmp.append(cwt_mngs)
                cwts.append(cwts_tmp)
            cwts = np.array(cwts)

            # i_trial = 0
            # i_ch = 0
            # nyq = int(SAMP_RATE_iEEG / 2)
            # cwt_mngs = mngs.dsp.wavelet(iEEG[i_trial, i_ch], SAMP_RATE_iEEG, f_min=0.1)
            # cwt_mngs = cwt_mngs[cwt_mngs.index < nyq]
            
            sdir = f"./data/Sub_{i_sub_str}/Session_{i_session_str}/"            
            mngs.io.save(cwts, f"{sdir}cwts.npy") # (50, 16, 92, 16000)
            mngs.io.save(np.array(cwt_mngs.index), f"{sdir}freqs.npy")

            # ./data/Sub_09/Session_02/cwts.npy

            """
            cwts = abs(cwts) / abs(np.array(cwts)).max(axis=-1, keepdims=True)

            time = np.arange(cwts.shape[-1]) / SAMP_RATE_iEEG

            cwt_mean_df = pd.DataFrame(
                data=cwts.reshape(-1, cwts.shape[-2], cwts.shape[-1]).mean(axis=0),
                index=np.array(cwt_mngs.index).round(3),
                columns=time,
                )

            cwt_std_df = pd.DataFrame(
                data=cwts.reshape(-1, cwts.shape[-2], cwts.shape[-1]).std(axis=0),
                index=np.array(cwt_mngs.index).round(3),
                columns=time,
                )

            # cwt.index = np.array(cwt.index).round(3)
            # cwt.columns = time
            cwt_mean_df = cwt_mean_df.loc[cwt_mean_df.index[::-1]]
            cwt_std_df = cwt_std_df.loc[cwt_std_df.index[::-1]]    
            # cwt = cwt.loc[cwt.index[::-1]]
            low_hz = 1
            high_hz = 500
            indi = (low_hz < cwt_mean_df.index) & (cwt_mean_df.index < high_hz)

            fig, ax = plt.subplots()    
            sns.heatmap(
                cwt_mean_df[indi],
                # cwt_std_df[indi],
                # cwt_mean_df,
                ax=ax,
                )
            plt.show()
            """

            # scale_factor = .125
            # dt = 1 / SAMP_RATE_iEEG

            # # wa = WaveletTransform(dt, scale_factor)    
            # wa = WaveletTransformTorch(dt, scale_factor, cuda=True)

            # power = wa.power(np.array(iEEG[i_trial]))
            # # cwt = wa.cwt(np.array(iEEG[i_trial]))
            # # cwt = abs(cwt) / abs(cwt).max(axis=1, keepdims=True).max(axis=2, keepdims=True)
            # freqs = wa.fourier_periods#.round(3)
            # time = np.arange(power.shape[-1]) / SAMP_RATE_iEEG

            # # plot_scalogram(power[0], wa.fourier_periods, time, ax=ax)
            # # plt.show()

            # freqs = np.exp(wa.fourier_periods)

            # # print(cwt.shape)

            # sns.heatmap(
            #     # abs(cwt[i_ch]) / abs(cwt[i_ch]).max(),
            #     # cwt[i_ch],
            #     # pd.DataFrame(data=cwt.mean(axis=0), columns=time, index=freqs),
            #     pd.DataFrame(data=power.mean(axis=0), columns=time, index=freqs),        
            #     ax=ax,
            #     )
            # plt.show()

