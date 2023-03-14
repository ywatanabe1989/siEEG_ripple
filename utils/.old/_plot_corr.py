#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-02-25 23:25:37 (ywatanabe)"


import mngs
import pandas as pd
import seaborn as sns

def plot_corr(corr_obs, corrs_shuffled, label="label"):
    fig, ax = plt.subplots()#figsize=(1.2, .84))
    data = pd.DataFrame({"correlation": corrs_shuffled})
    data = mngs.plt.add_hue(data)
    data["x"] = label
    sns.violinplot(data=data,
                   x="x",
                   y="correlation",
                   ax=ax,
                   hue="hue",
                   hue_order=[0,1],
                   split=True,
                   color="gray",
                   width=0.08,
                   linewidth=.5,
                   )
    ax.scatter(
        x=0,
        y=corr_obs,
        color="red",
        s=100,
        )
    ylim_val = 0.31
    ax.set_ylim(-ylim_val, ylim_val)
    ax = mngs.plt.ax_extend(ax, x_extend_ratio=0.3, y_extend_ratio=1.)
    
    # ax.set_xlim(-.1, 5)    
    ax.legend_ = None

    # plt.show()
    return fig


if __name__ == "__main__":
    import numpy as np
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    corr_obs = 0.3
    corrs_shuffled = .3*(np.random.rand(1000) - .5)

    fig = plot_corr(corr_obs, corrs_shuffled)

    plt.show()
