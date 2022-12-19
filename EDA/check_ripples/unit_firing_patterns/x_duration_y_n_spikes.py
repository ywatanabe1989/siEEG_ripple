#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-02 12:13:40 (ywatanabe)"

import mngs
import numpy as np

import mngs

import matplotlib.pyplot as plt

import sys

sys.path.append(".")
from eeg_ieeg_ripple_clf import utils
import seaborn as sns
import scipy

def plot_x_duration_y_n_spikes(rips_df):
    fig, axes = plt.subplots(ncols=4, sharex=True, sharey=True, figsize=(6.4*2, 4.8))
    x = "duration_ms"
    y = "n_firings"
    for ax, phase in zip(axes, PHASES):
        data = rips_df[rips_df.phase == phase]

        # correlation
        corr, pval = scipy.stats.pearsonr(data.duration_ms, data.n_firings)

        # regression line
        for is_correct in [True]:
            data_ic = data[data.correct == is_correct]
            n_ic = len(data_ic)
            m_ic, b_ic = np.polyfit(data_ic[x], data_ic[y], 1)
            corr_ic = np.corrcoef(data_ic[x], data_ic[y])[0,1].round(2)
            label_ic = f"n = {n_ic}; r = {corr:.2f}; p = {pval:.2f}"
            color_ic = "blue" if not is_correct else "black"
            ax.plot(data_ic[x], m_ic * data_ic[x] + b_ic, color=color_ic, label=label_ic)

        # scatter
        sns.scatterplot(
            data=data,
            x="duration_ms",
            y="n_firings",
            # hue="correct",
            # scatter_kws={"alpha": .5},
            alpha=0.5,
            ax=ax,
            # hue_order=[False, True],
            # scatter_kws={"hue_order": [False, True]},
        )

        ax.set_ylabel("")    
        ax.set_xlabel("")
        ax.legend()

    fig.supxlabel("Ripple duration [ms]")
    fig.supylabel("# of unit firings in the ripple event")
    return fig

if __name__ == "__main__":
    # Loads
    rips_df = utils.load_rips()
    PHASES = mngs.io.load("./config/global.yaml")["PHASES"]

    fig = plot_x_duration_y_n_spikes(rips_df)
    mngs.io.save(fig, "./tmp/figs/scatter/x_duration_y_n_spikes.png")    
