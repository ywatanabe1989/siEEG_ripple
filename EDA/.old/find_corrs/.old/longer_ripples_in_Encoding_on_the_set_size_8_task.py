#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-11-09 12:10:11 (ywatanabe)"

import mngs
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

for sd in [2., 3., 4.]:
    rips_df = mngs.io.load(f"./tmp/rips_df_bi_{sd}_SD.csv")
    rips_df["Ripple duration [ms]"] = (rips_df["end_time"] - rips_df["start_time"]) * 1000

    indi = rips_df["session"] <= 2
    rips_df = rips_df[indi]

    fig, axes = plt.subplots(ncols=2, sharey=True)

    for i_ax, ax in enumerate(axes):
        is_correct = bool(i_ax)
        indi = rips_df["correct"] == is_correct

        y = "Ripple duration [ms]"
        data = rips_df[indi]
        data = data[~data[y].isna()]

        sns.boxplot(
            data=data,
            y=y,
            x="set_size",
            hue="phase",
            hue_order=["Fixation", "Encoding", "Maintenance", "Retrieval"],
            ax=ax,
            )
        correct_str = "Correct" if is_correct else "Incorrect"
        ax.set_title(correct_str)
    mngs.io.save(fig, f"./tmp/longer_ripples_in_Encoding_on_the_set_size_8_task_{sd}_SD.png")
    # plt.show()
