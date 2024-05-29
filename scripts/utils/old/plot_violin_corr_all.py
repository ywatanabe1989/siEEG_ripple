#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-04-12 11:22:32 (ywatanabe)"

import mngs
import pandas as pd
import seaborn as sns

def plot_violin(fpaths):

    df = pd.DataFrame()
    for fpath in fpaths:
        fname = fpath.split("/")[-1].split(".")[0]
        data = mngs.io.load(fpath)
        # print(bisect_right(sorted(np.array(data["surrogate"])), data["observed"]))
        df_obs = pd.DataFrame(
            pd.Series({"correlation": data["observed"]}, name="correlation")
        )
        print(float(df_obs.round(3).iloc[0]))
        df_sur = pd.DataFrame({"correlation": np.array(data["surrogate"]).squeeze()})
        _df = pd.concat([df_obs, df_sur])
        _df["is_obs"] = [True] + [False for _ in range(len(df_sur))]
        _df["variable"] = fname
        df = pd.concat([df, _df])

    n_violins = len(df.variable.unique())

    df = mngs.plt.add_hue(df)
    fig, ax = plt.subplots(figsize=(6.4*(n_violins//2), 4.8))
    sns.violinplot(
        data=df[df.is_obs == False],
        x="variable",
        y="correlation",
        hue="hue",
        hue_order=[0, 1],
        split=True,
        color="gray",
        ax=ax,
        width=.5,
    )
    for _, row_obs in df[df.is_obs == True].iterrows():
        ax.scatter(
            x=row_obs["variable"],
            y=row_obs["correlation"],
            color="red",
            s=100,
        )
    ylim = 0.27
    ax.set_ylim(-ylim, ylim)
    # ax = mngs.plt.ax_extend(ax, x_extend_ratio=3, y_extend_ratio=1)
    plt.xticks(rotation=90)
    plt.show()

    # mngs.io.save(fig, "./tmp/figs/corr/violin_all.tif")
    return fig

def print_rank(fpaths):
    for fpath in fpaths:
        print(fpath)
        data = mngs.io.load(fpath)
        print(bisect_right(np.array(data["surrogate"]), data["observed"]))
        print()

if __name__ == "__main__":
    import matplotlib
    import numpy as np

    matplotlib.use("TkAgg")
    from glob import glob

    import matplotlib.pyplot as plt
    from natsort import natsorted
    from bisect import bisect_right

    # Loads
    fpaths = natsorted(glob("./res/figs/corr/*"))

    # Plots
    fig = plot_violin(fpaths)
    mngs.io.save(fig, "./res/figs/violin/corrs_all.tiff")
    plt.show()

    # Checks the rank
    print_rank(fpaths)

