#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-03-12 12:55:13 (ywatanabe)"

import mngs
import pandas as pd
import seaborn as sns


def print_ranks(fpaths):
    from bisect import bisect_right

    for fpath in fpaths:
        print(fpath)
        data = mngs.io.load(fpath)
        rank = bisect_right(np.array(data["surrogate"]), data["observed"])
        
        mid_rank = len(data["surrogate"]) // 2

        if rank <= mid_rank:
            rank /= 2
        if mid_rank < rank:
            rank = (len(data["surrogate"]) - rank) // 2            
        pval = rank / len(data["surrogate"])

        mark = mngs.stats.to_asterisks(pval)        

        # if not only_significant:
        #     print(round(corr_obs, 2), round(pval, 3), mark)        
        # if only_significant and (pval < 0.05):
        print(round(data["observed"], 2), round(pval, 3), mark)                
        


def to_df(fpaths):
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
    df = mngs.plt.add_hue(df)
    return df


def plot_violin(df, ylim=0.3):
    n_violins = len(df["variable"].unique()) - 1
    width = 6.4 / 4 * n_violins
    fig, ax = plt.subplots(figsize=(width, 4.8))  # 6.4*4

    sns.violinplot(
        data=df[df.is_obs == False],
        x="variable",
        # order=natsorted(df["variable"].unique()[:-1]),
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
        # ax.legend(visible=False)
    plt.legend().remove()
    # ylim = 0.22
    ax.set_ylim(-ylim, ylim)
    # ax = mngs.plt.ax_extend(ax, x_extend_ratio=3, y_extend_ratio=1)
    plt.xticks(rotation=90)
    # fig._legend = None
    return fig

def main(dir):
    fpaths = natsorted(glob(dir + "*.pkl"))
    print_ranks(fpaths)

    df = to_df(fpaths)

    fig = plot_violin(df, ylim=.6)
    # plt.show()

    mngs.io.save(fig, dir + "violin.tif")

if __name__ == "__main__":
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    from glob import glob

    import matplotlib.pyplot as plt
    from natsort import natsorted

    for var in ["dist_from_O", "speed"]:
        for match in [1,2]:
            dir = f"./tmp/figs/corr/peri_SWR_{var}/match_{match}/"
            main(dir)
            import ipdb; ipdb.set_trace()
