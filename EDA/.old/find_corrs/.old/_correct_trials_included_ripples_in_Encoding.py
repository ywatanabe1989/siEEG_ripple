#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-11-10 13:05:13 (ywatanabe)"

import mngs
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from natsort import natsorted

iEEG_positions_str = "PHL" # "" # 14832
# iEEG_positions_str = "ECL_ECR" # "" # 14832
rips_df = mngs.io.load(f"./tmp/rips_df_bi_2.0_SD_{iEEG_positions_str}.csv")

# iEEG_positions_str = "" # "" # 14832
# rips_df = mngs.io.load(f"./tmp/rips_df_bi_2.0_SD.csv") # 14762


rips_df["Ripple duration [ms]"] = (rips_df["end_time"] - rips_df["start_time"]) * 1000
rips_df["Ripple peak amplitude [SD of baseline]"] = rips_df['ripple_peak_amplitude_sd']

rips_df = rips_df[rips_df["session"] <= 2]
rips_df = rips_df[rips_df["set_size"] == 8]


# undersampling
indi_incorrect = rips_df["correct"] == False
n_incorrect = (indi_incorrect).sum()


indi_correct = natsorted(np.random.permutation((rips_df["correct"] == True).index)[:n_incorrect])

rips_df = pd.concat([rips_df[indi_incorrect], rips_df.loc[indi_correct]]).sort_index()
n_correct = (rips_df["correct"] == True).sum()



for phase in ["Fixation", "Encoding", "Maintenance", "Retrieval"]:
    
    indi_1 = rips_df["phase"] == phase

    fig, axes = plt.subplots(ncols=2, sharey=True, sharex=True)

    for i_ax, ax in enumerate(axes):
        is_correct = bool(i_ax)
        indi_2 = indi_1 * (rips_df["correct"] == is_correct)

        sns.scatterplot(
            data=rips_df[indi_2],
            x="Ripple duration [ms]",
            y="Ripple peak amplitude [SD of baseline]",
            hue="set_size",
            alpha=.3,
            ax=ax,
            )
        ax.set_xlim(0, 600)
        correct_str = "Correct" if is_correct else "Incorrect"
        ax.set_title(correct_str)
        fig.suptitle(phase)
    mngs.io.save(fig, f"./tmp/figs/{iEEG_positions_str}_ripples_scatter_in_{phase}.png")
    plt.close()
