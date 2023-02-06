#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-01-25 23:24:20 (ywatanabe)"

import mngs
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression, RANSACRegressor
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

## Functions
def calc_corr(col):
    X = np.array(dfs["count"]).reshape(-1, 1)

    # y = np.log10(dfs[col])

    y = dfs[col]

    corr = np.corrcoef(X.squeeze(), y)[0, 1]  # fixme

    shuffled_corrs = []
    for _ in range(1000):
        X_shuffle = np.random.permutation(X)
        # lr_shuffle = LinearRegression().fit(X_shuffle,y)
        # shuffled_corr = lr_shuffle.corr(X_shuffle,y)
        shuffled_corr = np.corrcoef(X_shuffle.squeeze(), y)[0, 1]  # fixme
        shuffled_corrs.append(shuffled_corr)
    shuffled_corrs = pd.DataFrame(shuffled_corrs)

    corr_0950 = shuffled_corrs.quantile(0.950).iloc[0]
    corr_0990 = shuffled_corrs.quantile(0.990).iloc[0]
    corr_0999 = shuffled_corrs.quantile(0.999).iloc[0]

    sign_level = "p < 0.05" if corr_0950 < corr else False
    sign_level = "p < 0.01" if corr_0990 < corr else sign_level
    sign_level = "p < 0.001" if corr_0999 < corr else sign_level

    print(sign_level, corr_0950, corr_0990, corr_0999, corr)

    return corr, np.array(shuffled_corrs).squeeze()

if __name__ == "__main__":
    
    mngs.general.fix_seeds(42, np=np)

    # count: 0 -> set_size 4, Match_IN
    # count: 1 -> set_size 4, Mismatch_OUT
    # count: 2 -> set_size 6, Match_IN
    # count: 3 -> set_size 6, Mismatch_OUT
    # count: 4 -> set_size 8, Match_IN
    # count: 5 -> set_size 8, Mismatch_OUT
    dfs = pd.DataFrame()
    count = 0
    for set_size in [4, 6, 8]:
        for match in [1, 2]:
            lpath = (
                f"./tmp/g_dist_z_by_session/Hipp._{set_size}_match_{match}_raw_dists.csv"
            )
            df = mngs.io.load(lpath)
            df["set_size"] = set_size
            df["match"] = match
            df["count"] = count
            count += 1
            dfs = pd.concat([dfs, df])

    # to df
    shuffled_corrs_all = pd.DataFrame()
    for phase_combi in ["FE", "FM", "FR", "EM", "ER", "MR"]:
        corr, shuffled_corrs = calc_corr(phase_combi)

        pc = np.array([phase_combi for _ in range(len(shuffled_corrs) + 1)])
        corrs = np.hstack([shuffled_corrs, np.array(corr)])
        is_shuffled = np.hstack(
            [np.array([True for _ in range(len(shuffled_corrs))]), np.array(False)]
        )

        _df = pd.DataFrame(
            {
                "phase_combi": pc,
                "is_shuffled": is_shuffled,
                "corrs": corrs,
            }
        )

        shuffled_corrs_all = pd.concat([shuffled_corrs_all, _df])

    # add hue to split violinplot
    shuffled_corrs_all["hue"] = 0
    _df = pd.DataFrame(shuffled_corrs_all.iloc[-1]).T
    _df["phase_combi"], _df["is_shuffled"], _df["corrs"], _df["hue"] = None, None, np.nan, 1
    shuffled_corrs_all = pd.concat([shuffled_corrs_all, _df])

    # plots
    fig, ax = plt.subplots()
    sns.violinplot(
        data=shuffled_corrs_all,
        x="phase_combi",
        y="corrs",
        order=["FE", "FM", "FR", "EM", "ER", "MR"],
        hue="hue",
        hue_order=[0, 1],
        split=True,
        alpha=0.3,
        ax=ax,
        color="black",
        legend=False,
        width=0.4,
    )
    ax.scatter(
        x=shuffled_corrs_all[shuffled_corrs_all.is_shuffled == False]["phase_combi"],
        y=shuffled_corrs_all[shuffled_corrs_all.is_shuffled == False]["corrs"],
        s=100,
        color=mngs.plt.colors.to_RGBA("red", alpha=1),
    )
    plt.legend([], [], frameon=False)
    ax.set_ylim(-0.15, 0.15)
    # ax.set_ylim(-0.001, 0.014)
    mngs.io.save(fig, "./tmp/figs/violin/linear_regression_corr.tif")
    plt.show()
