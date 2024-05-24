#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-04-12 10:40:25 (ywatanabe)"

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
from itertools import combinations
import scipy
import statsmodels as sm
import scikit_posthocs as sp

## Functions
def load_log10_g_dists_Hipp(match):
    # to df
    dfs = pd.DataFrame()
    # count = 0
    for set_size in [4, 6, 8]:
        lpath = (
            f"./tmp/g_dist_z_by_session/Hipp._{set_size}_match_{match}_raw_dists.csv"
        )
        df = mngs.io.load(lpath)
        del df["Unnamed: 0"]
        df = np.log10(df)
        df["set_size"] = set_size
        df["match"] = match
        # df["count"] = count
        # count += 1
        dfs = pd.concat([dfs, df])
    return dfs


def test_kw_and_bm(dfs):

    for col in dfs.columns[:6]:
        print(col)

        df_col = dfs[[col, "set_size", "match"]]

        # Kruskal-Wallis
        # df_col_sm = {
        #     f"set_size_{set_size}_match_{match}": df_col[
        #         (df_col.set_size == set_size) * (df_col.match == match)
        #     ]
        #     for set_size in [4, 6, 8]
        # }
        df_col_sm = {
            f"set_size_{set_size}": df_col[
                (df_col.set_size == set_size)
            ]
            for set_size in [4, 6, 8]
        }

        stats, pval_kw = scipy.stats.kruskal(
            *[_df.iloc[:, 0] for _df in df_col_sm.values()]
        )

        if pval_kw < 0.05:
            print(f"Kruskal Wallis ss m: {round(pval_kw, 3)}")

        # Post-hoc BM
        nn_pair = len(list(combinations(df_col_sm.values(), 2)))
        for df1, df2 in combinations(df_col_sm.values(), 2):
            w, pval_bm, dof, effsize = mngs.stats.brunner_munzel_test(
                df1.iloc[:, 0], df2.iloc[:, 0]
            )
            pval_bm *= nn_pair  # Bonferroni correction
            if pval_bm < 0.05:
                print(df1.set_size.iloc[0], df2.set_size.iloc[0])
                print(f"Brunner-Munzel (corrected): {pval_bm}")

        print()
        # import ipdb; ipdb.set_trace()


def plot_shuffled_corrs(shuffled_corrs_all):
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
    ylim_val = 0.21
    ax.set_ylim(-ylim_val, ylim_val)
    # ax.set_ylim(-0.001, 0.014)
    plt.show()
    return fig


def corr_test(df_log10_g_dists):
    phase_combi_all = ["FE", "FM", "FR", "EM", "ER", "MR"]
    out = {}
    for i_pc, phase_combi in enumerate(phase_combi_all):
        print(phase_combi)
        pval, corr_obs, corrs_shuffled = mngs.stats.corr_test(
            df_log10_g_dists[phase_combi], df_log10_g_dists["set_size"]
        )
        out[phase_combi] = {"observed": corr_obs, "surrogate": corrs_shuffled}
    return out


if __name__ == "__main__":

    mngs.general.fix_seeds(42, np=np)

    df_log10_g_dists = load_log10_g_dists_Hipp(None)
    test_kw_and_bm(df_log10_g_dists)
    corr_out = corr_test(df_log10_g_dists)
    for ii, (key, val) in enumerate(corr_out.items()):
        mngs.io.save(val, f"./res/figs/corr/{ii+3:02d}_{key}.pkl")

    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.boxplot(
        data=df_log10_g_dists[["ER", "set_size"]],
        x="set_size",
        y="ER",
        showfliers=False,
        )
    plt.show()
    
    # for match in [1, 2]:

    #     # df_log10_g_dists = load_log10_g_dists_Hipp(match)
    #     df_log10_g_dists = load_log10_g_dists_Hipp(None)        

    #     # for col in df_log10_g_dists.columns[:6]:
    #     #     w, p, dof, eff = mngs.stats.brunner_munzel_test(
    #     #         df_log10_g_dists[col][df_log10_g_dists.set_size == 6],
    #     #         df_log10_g_dists[col][df_log10_g_dists.set_size == 8],
    #     #         )
    #     #     print(col)
    #     #     print(p)

    #     test_kw_and_bm(df_log10_g_dists)
    #     corr_test(df_log10_g_dists)
    #     # shuffled_corrs_all = corr_test(df_log10_g_dists)

    #     # fig = plot_shuffled_corrs(shuffled_corrs_all)
    #     # mngs.io.save(fig, f"./tmp/figs/violin/corr_between_set_size_and_dist_match_{match}.tif")
