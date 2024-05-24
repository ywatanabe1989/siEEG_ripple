#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-04-14 11:15:42 (ywatanabe)"

# from elephant.gpfa import GPFA
# import quantities as pq
# import mngs
# import neo
# import numpy as np
# import pandas as pd
# from numpy.linalg import norm
# import mngs
# import matplotlib

# matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt

# import seaborn as sns
# import random
# from scipy.stats import brunnermunzel
# from itertools import product, combinations
# import warnings

import sys

sys.path.append("./siEEG_ripple")
import utils

import mngs
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from itertools import combinations

# Functions
def slice_events_df(events_df, subject, session, phase, set_size):
    indi = events_df.subject == subject
    indi *= events_df.session == session
    indi *= events_df.phase == phase
    indi *= events_df.set_size == set_size
    return events_df[indi]


# Get gs
def calc_dists_period_ss(gs_session, period, set_size):
    if period is not None:
        _gs_session_period_ss = gs_session[
            (gs_session.index == f"g_{period}") * (gs_session.set_size == set_size)
        ]
    else:
        _gs_session_period_ss = gs_session[(gs_session.set_size == set_size)]

    dists = {}
    for p1, p2 in combinations(PHASES, 2):
        dist = mngs.linalg.nannorm(
            np.array(
                (_gs_session_period_ss[_gs_session_period_ss.phase == p1])[[0, 1, 2]]
            )
            - np.array(
                (_gs_session_period_ss[_gs_session_period_ss.phase == p2])[[0, 1, 2]]
            )
        )
        dists[p1[0] + p2[0]] = dist
    try:
        df = pd.DataFrame(dists)
    except:
        df = pd.DataFrame(pd.Series(dists)).T
    df["period"] = period
    df["set_size"] = set_size
    return df


def calc_dists(events_df):
    dists_df_all = pd.DataFrame()
    for subject in events_df["subject"].unique():
        for session in events_df["session"].unique():
            dists_df = pd.DataFrame()
            gs_session = pd.DataFrame()
            for phase in PHASES:
                for set_size in [4, 6, 8]:
                    _events_df = slice_events_df(
                        events_df, subject, session, phase, set_size
                    )

                    coords_pre = pd.concat(
                        [_events_df[f"{ii}"] for ii in SWR_BINS["pre"]]
                    )
                    coords_mid = pd.concat(
                        [_events_df[f"{ii}"] for ii in SWR_BINS["mid"]]
                    )
                    coords_post = pd.concat(
                        [_events_df[f"{ii}"] for ii in SWR_BINS["post"]]
                    )

                    try:
                        g_pre = np.nanmedian(np.vstack(coords_pre), axis=0)
                        g_mid = np.nanmedian(np.vstack(coords_mid), axis=0)
                        g_post = np.nanmedian(np.vstack(coords_post), axis=0)
                        g_none = np.nanmedian(
                            np.vstack(
                                [
                                    np.vstack(coords_pre),
                                    np.vstack(coords_mid),
                                    np.vstack(coords_post),
                                ]
                            ),
                            axis=0,
                        )

                    except Exception as e:
                        print(e)
                        g_pre = np.array([np.nan, np.nan, np.nan])
                        g_mid = np.array([np.nan, np.nan, np.nan])
                        g_post = np.array([np.nan, np.nan, np.nan])
                        g_none = np.array([np.nan, np.nan, np.nan])

                    gs_df = pd.DataFrame(
                        {
                            "g_pre": g_pre,
                            "g_mid": g_mid,
                            "g_post": g_post,
                            "g_none": g_none,
                        }
                    ).T

                    gs_df["phase"] = phase
                    gs_df["set_size"] = set_size
                    gs_session = pd.concat([gs_session, gs_df])

            for period in ["pre", "mid", "post", "none"]:
                for set_size in [4, 6, 8]:
                    dists_period_ss = calc_dists_period_ss(gs_session, period, set_size)
                    dists_df = pd.concat([dists_df, dists_period_ss])
                    dists_df["subject"] = subject
                    dists_df["session"] = session
            dists_df_all = pd.concat([dists_df_all, dists_df])
            # dists_pre =
            # dists_pre = calc_dists(gs_session, "pre", 4)
            # dists_pre = calc_dists(gs_session, "pre", 4)
            # dists_pre = calc_dists(gs_session, "pre", 4)
            # dists_pre = calc_dists(gs_session, "pre", 4)
            # dists_pre = calc_dists(gs_session, "pre", 4)
            # dists_pre = calc_dists(gs_session, "pre", 4)

    return dists_df_all[~dists_df_all.isna().any(axis=1)]

def print_corr(df):
    for period in ["pre", "mid", "post", "none"]:
        print(period)
        df_period = df[df.period == period]
        pval, corr_obs, corrs_shuffled = mngs.stats.corr_test(
            np.log10(df_period["FE"]), df_period["set_size"]
        )
        pval, corr_obs, corrs_shuffled = mngs.stats.corr_test(
            np.log10(df_period["FM"]), df_period["set_size"]
        )
        pval, corr_obs, corrs_shuffled = mngs.stats.corr_test(
            np.log10(df_period["FR"]), df_period["set_size"]
        )
        pval, corr_obs, corrs_shuffled = mngs.stats.corr_test(
            np.log10(df_period["EM"]), df_period["set_size"]
        )
        pval, corr_obs, corrs_shuffled = mngs.stats.corr_test(
            np.log10(df_period["ER"]), df_period["set_size"]
        )
        pval, corr_obs, corrs_shuffled = mngs.stats.corr_test(
            np.log10(df_period["MR"]), df_period["set_size"]
        )
        print()


if __name__ == "__main__":
    (
        PHASES,
        PHASES_BINS_DICT,
        GS_BINS_DICT,
        COLORS_DICT,
        BIN_SIZE,
    ) = utils.define_phase_time()
    SWR_BINS = mngs.io.load("./config/global.yaml")["SWR_BINS"]
    # rips_df = utils.rips.add_coordinates(
    #     utils.rips.load_rips_df_with_traj(BIN_SIZE, is_control=False)
    # )
    # cons_df = utils.rips.add_coordinates(
    #     utils.rips.load_rips_df_with_traj(BIN_SIZE, is_control=True)
    # )

    dists_rips = calc_dists(rips_df)
    dists_cons = calc_dists(cons_df)    
    # df[df.period == "none"]

    # import seaborn as sns

    # fig, ax = plt.subplots()
    # for i_col, col in enumerate(df.columns[:6]):
    #     ax.boxplot(df[df.period=="none"][col], positions=[i_col])
    # plt.show()

    print_corr(dists_rips)
    # pre
    # Corr. = 0.21; p-val = 0.074 ns
    # Corr. = 0.04; p-val = 0.236 ns
    # Corr. = 0.24; p-val = 0.074 ns
    # Corr. = 0.26; p-val = 0.003 **
    # Corr. = 0.09; p-val = 0.104 ns
    # Corr. = 0.1; p-val = 0.098 ns

    # mid
    # Corr. = 0.2; p-val = 0.003 **
    # Corr. = 0.36; p-val = 0.003 **
    # Corr. = 0.25; p-val = 0.003 **
    # Corr. = 0.27; p-val = 0.032 *
    # Corr. = 0.34; p-val = 0.0 ***
    # Corr. = 0.17; p-val = 0.004 **

    # post
    # Corr. = 0.33; p-val = 0.0 ***
    # Corr. = 0.22; p-val = 0.074 ns
    # Corr. = 0.15; p-val = 0.074 ns
    # Corr. = 0.21; p-val = 0.125 ns
    # Corr. = 0.45; p-val = 0.015 *
    # Corr. = 0.16; p-val = 0.125 ns

    # none
    # Corr. = 0.25; p-val = 0.003 **
    # Corr. = 0.15; p-val = 0.093 ns
    # Corr. = 0.2; p-val = 0.062 ns
    # Corr. = 0.31; p-val = 0.032 *
    # Corr. = 0.3; p-val = 0.0 ***
    # Corr. = 0.09; p-val = 0.227 ns
    print_corr(dists_cons)
    # pre
    # Corr. = 0.36; p-val = 0.0 ***
    # Corr. = 0.3; p-val = 0.126 ns
    # Corr. = 0.21; p-val = 0.007 **
    # Corr. = 0.03; p-val = 0.098 ns
    # Corr. = 0.26; p-val = 0.0 ***
    # Corr. = 0.33; p-val = 0.007 **

    # mid
    # Corr. = 0.08; p-val = 0.039 *
    # Corr. = 0.21; p-val = 0.125 ns
    # Corr. = 0.07; p-val = 0.039 *
    # Corr. = -0.08; p-val = 0.154 ns
    # Corr. = 0.29; p-val = 0.0 ***
    # Corr. = 0.08; p-val = 0.039 *

    # post
    # Corr. = -0.04; p-val = 0.008 **
    # Corr. = 0.07; p-val = 0.039 *
    # Corr. = -0.0; p-val = 0.056 ns
    # Corr. = 0.15; p-val = 0.038 *
    # Corr. = 0.14; p-val = 0.053 ns
    # Corr. = 0.05; p-val = 0.039 *

    # none
    # Corr. = 0.09; p-val = 0.005 **
    # Corr. = 0.18; p-val = 0.003 **
    # Corr. = 0.02; p-val = 0.008 **
    # Corr. = 0.04; p-val = 0.137 ns
    # Corr. = 0.24; p-val = 0.0 ***
    # Corr. = 0.11; p-val = 0.039 *

    
