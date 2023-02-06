#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-11-09 16:21:48 (ywatanabe)"

import mngs
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

rips_df = mngs.io.load("./tmp/rips_df_bi_2.0_SD.csv")
rips_df["Ripple duration [ms]"] = (rips_df["end_time"] - rips_df["start_time"]) * 1000
rips_df["Ripple peak amplitude [SD of baseline]"] = rips_df['ripple_peak_amplitude_sd']

indi = rips_df["session"] <= 2

for phase in ["Fixation", "Encoding", "Maintenance", "Retrieval"]:
    indi2 = indi * rips_df["phase"] == phase

    fig, axes = plt.subplots(ncols=2, sharey=True, sharex=True)

    for i_ax, ax in enumerate(axes):
        is_correct = bool(i_ax)
        indi3 = indi2 * (rips_df["correct"] == is_correct)

        sns.scatterplot(
            data=rips_df[indi3],
            x="Ripple duration [ms]",
            y="Ripple peak amplitude [SD of baseline]",
            # y='ripple_amplitude_sd',    
            hue="set_size",
            # hue_order=["Fixation", "Encoding", "Maintenance", "Retrieval"],
            alpha=.3,
            ax=ax,
            )
        correct_str = "Correct" if is_correct else "Incorrect"
        ax.set_title(correct_str)
        fig.suptitle(phase)
    mngs.io.save(fig, f"./tmp/correct_trials_included_ripples_in_{phase}.png")
