#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-02-18 09:26:49 (ywatanabe)"

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
def load_data(match):
    # to df    
    dfs = pd.DataFrame()
    count = 0
    for set_size in [4, 6, 8]:
        lpath = (
            f"./tmp/g_dist_z_by_session/Hipp._{set_size}_match_{match}_raw_dists.csv"
        )
        df = mngs.io.load(lpath)
        df["set_size"] = set_size
        df["match"] = match
        df["count"] = count
        count += 1
        dfs = pd.concat([dfs, df])
    return dfs

def calc_corr(dfs, col):
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

def test_kw_and_bm(dfs):
    # Kruskal-Wallis
    for col in dfs.columns[1:7]:
        print(col)

        df_col = dfs[[col, "set_size", "match"]]

        df_col_ss_m = {}
        for set_size in [4, 6, 8]:
            df_col_ss_m[f"set_size_{set_size}_match_{match}"] = \
                df_col[(df_col.set_size == set_size) * (df_col.match == match)]
        # dict_keys(['set_size_4_match_1', 'set_size_6_match_1', 'set_size_8_match_1'])
                
        stats, pval_kw = scipy.stats.kruskal(*[_df.iloc[:,0] for _df in df_col_ss_m.values()])
        print(f"Kruskal Wallis ss m: {round(pval_kw, 3)}")


        # # Post-hoc Conover
        # fdf = {}
        # for _col, _df in df_col_ss_m.items():
        #     fdf[_col] = _df.iloc[:,0]
        # fdf = np.array(mngs.gen.force_dataframe(fdf).replace({'': np.nan})).T
        # print(f"Posthoc Conover: {sp.posthoc_conover(fdf)}")

        nn_pair = len(list(combinations(df_col_ss_m.values(), 2)))
        for df1, df2 in combinations(df_col_ss_m.values(), 2):
            
            # print(df1.set_size.iloc[0], df1.match.iloc[0], df2.set_size.iloc[0], df2.match.iloc[0])
            print(df1.set_size.iloc[0], df2.set_size.iloc[0])

            # print(mngs.gen.describe(np.array(df1.iloc[:,0]), method="median"))
            # print(mngs.gen.describe(np.array(df2.iloc[:,0]), method="median"))           
            w, pval_bm, dof, effsize = mngs.stats.brunner_munzel_test(df1.iloc[:,0], df2.iloc[:,0])
            pval_bm *= nn_pair # Bonferroni correction
            
            print(f"Brunner-Munzel (corrected): {pval_bm}")
            print()
        # import ipdb; ipdb.set_trace()

def corr_test(dfs):
    shuffled_corrs_all = pd.DataFrame()
    for phase_combi in ["FE", "FM", "FR", "EM", "ER", "MR"]:
        print(phase_combi)
        corr, shuffled_corrs = calc_corr(dfs, phase_combi)
        # print(phase_combi, corr, mngs.gen.describe(shuffled_corrs, method="median"))

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
    return shuffled_corrs_all

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
    ylim_val = .21
    ax.set_ylim(-ylim_val, ylim_val)
    # ax.set_ylim(-0.001, 0.014)
    plt.show()
    return fig

if __name__ == "__main__":
    
    mngs.general.fix_seeds(42, np=np)

    for match in [1,2]:

        dfs = load_data(match)

        test_kw_and_bm(dfs)
        shuffled_corrs_all = corr_test(dfs)

        fig = plot_shuffled_corrs(shuffled_corrs_all)
        mngs.io.save(fig, f"./tmp/figs/violin/corr_between_set_size_and_dist_match_{match}.tif")        
