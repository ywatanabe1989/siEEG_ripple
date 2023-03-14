#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-03-13 13:07:45 (ywatanabe)"

"""
./tmp/figs/time_dependent_dist
./tmp/figs/hist/traj_dist
./tmp/figs/hist/traj_pos
"""

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from elephant.gpfa import GPFA
import mngs
import neo
import numpy as np
import pandas as pd
import ffmpeg
from matplotlib import animation
import os
from mpl_toolkits.mplot3d import Axes3D
from bisect import bisect_right
from numpy.linalg import norm
import seaborn as sns
import random
from scipy.stats import brunnermunzel, ttest_ind
from scipy.linalg import norm
import sys
import quantities as pq

sys.path.append(".")
import utils
from itertools import combinations, product
import seaborn as sns
from scipy.stats import kruskal

# Functions

def get_speed(subject, session, roi):
    speeds_all = []
    subject = f"{int(subject):02d}"
    traj_session = mngs.io.load(
        f"./data/Sub_{subject}/Session_{session}/traj_z_by_session_{roi}.npy"
    )
    speed_session = traj_session[..., 1:] - traj_session[..., :-1]
    return speed_session


def add_speed_events(events_df):
    def get_event_speed(event):
        speed_session = get_speed(event.subject, event.session, event.ROI)
        i_trial = int(event.trial_number - 1)
        center_bin = int(event.center_time / (50 / 1000))
        speed_event = speed_session[i_trial, :, center_bin - 2 : center_bin + 2].sum(
            axis=-1
        )
        return speed_event

    speeds_events = []
    for i_event, (_, event) in enumerate(events_df.iterrows()):
        speeds_events.append(get_event_speed(event))
    events_df["speed"] = speeds_events
    return events_df


def calc_corr(df_cosines_melted):
    for p1, p2 in combinations(PHASES, 2):
        print("-" * 40)
        print(p1, p2)

        data = []
        for set_size in [4,6,8]:
            data_ss = df_cosines_melted[
                        df_cosines_melted.variable == f"{p1[0]}{p2[0]}_{set_size}"
            ]
            data_ss["set_size"] = set_size
            data.append(data_ss)
        data = pd.concat(data)

        pval, corr_obs, corrs_sur = mngs.stats.corr_test(data.value, data.set_size)


def test_kw_and_bm(df_cosines_melted):
    for p1, p2 in combinations(PHASES, 2):
        print("-" * 40)
        print(p1, p2)
        stats, pval_kw = kruskal(
                df_cosines_melted[
                    df_cosines_melted.variable == f"{p1[0]}{p2[0]}_4"
                ].value,
                df_cosines_melted[
                    df_cosines_melted.variable == f"{p1[0]}{p2[0]}_6"
                ].value,
                df_cosines_melted[
                    df_cosines_melted.variable == f"{p1[0]}{p2[0]}_8"
                ].value,
            )
        if pval_kw < 0.05:
            print(f"Kruskal-Wallis: {round(pval_kw,3)}")

        for ss1, ss2 in combinations([4, 6, 8], 2):
            w, p, d, eff = mngs.stats.brunner_munzel_test(
                df_cosines_melted[
                    df_cosines_melted.variable == f"{p1[0]}{p2[0]}_{ss1}"
                ].value,
                df_cosines_melted[
                    df_cosines_melted.variable == f"{p1[0]}{p2[0]}_{ss2}"
                ].value,
            )
            if p*3 < 0.05:
                print(round(p*3, 3), ss1, ss2)
            
        print("-" * 40)
        print()



def calc_df_cosines(event_df, match):
    cosines_all = mngs.gen.listed_dict()
    for set_size in [4, 6, 8]:
        for subject, roi in ROIs.items():
            for session in ["01", "02"]:
                event_df_session = event_df[
                    (event_df.subject == f"{int(subject):02d}")
                    * (event_df.session == session)
                    * (event_df.match == match)
                    * (event_df.set_size == set_size)
                ][["speed", "phase", "set_size"]]
                for phase_1, phase_2 in combinations(PHASES, 2):

                    df_phase_1 = event_df_session[event_df_session.phase == phase_1]
                    df_phase_2 = event_df_session[event_df_session.phase == phase_2]

                    for _, row_1 in df_phase_1.iterrows():
                        for _, row_2 in df_phase_2.iterrows():
                            if row_1.set_size == row_2.set_size:
                                cosines_all[
                                    f"{phase_1[0]}{phase_2[0]}_{int(row_1.set_size)}"
                                ].append(
                                    mngs.linalg.cosine(row_1.speed, row_2.speed)
                                )

    df_cosines = mngs.gen.force_dataframe(cosines_all)
    df_cosines_melted = df_cosines.melt()
    df_cosines_melted = df_cosines_melted[df_cosines_melted.value != ""]
    df_cosines_melted["value"] = df_cosines_melted["value"].astype(float)
    df_cosines_melted = df_cosines_melted.sort_values("variable")
    return df_cosines_melted
        
def plot(ax, df_cosines_melted):
    order = [
        "FE_4",
        "FE_6",
        "FE_8",
        
        "FM_4",
        "FM_6",
        "FM_8",
        
        "FR_4",
        "FR_6",
        "FR_8",
        
        "EM_4",
        "EM_6",
        "EM_8",
        
        "ER_4",
        "ER_6",
        "ER_8",
        
        "MR_4",
        "MR_6",
        "MR_8",
    ]

    # order = ["FE",
    #          "FM",
    #          "FR",
    #          "EM",
    #          "ER",
    #          "MR",
    #          ]
    sns.boxplot(
        data=df_cosines_melted,
        x="variable",
        y="value",
        ax=ax,
        order=order,
        color="gray",
    )
    sns.stripplot(
        data=df_cosines_melted,
        x="variable",
        y="value",
        ax=ax,
        order=order,
        color="black",
    )
    ax.set_title(f"{match}")
    # sns.barplot(data=df_cosines_melted, x="variable", y="value", ax=ax)
    # plt.xticks(rotation=90)
    # ax.set_yscale("log")
    return ax, df_cosines_melted


if __name__ == "__main__":
    import mngs
    import numpy as np

    # Parameters
    (
        PHASES,
        PHASES_BINS_DICT,
        GS_BINS_DICT,
        COLORS_DICT,
        BIN_SIZE,
    ) = utils.define_phase_time()

    ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")
    rips_df = utils.rips.add_coordinates(
        utils.rips.load_rips_df_with_traj(BIN_SIZE, is_control=False)
    )
    center_bins_rips = (rips_df.center_time / (50 / 1000)).astype(int)
    indi_rips = pd.concat(
        [
            (ss <= center_bins_rips) * (center_bins_rips < ee)
            for ss, ee in GS_BINS_DICT.values()
        ],
        axis=1,
    ).sum(axis=1)
    rips_df = rips_df[indi_rips.astype(bool)]

    cons_df = utils.rips.add_coordinates(
        utils.rips.load_rips_df_with_traj(BIN_SIZE, is_control=True)
    )
    center_bins_cons = (cons_df.center_time / (50 / 1000)).astype(int)
    indi_cons = pd.concat(
        [
            (ss <= center_bins_cons) * (center_bins_cons < ee)
            for ss, ee in GS_BINS_DICT.values()
        ],
        axis=1,
    ).sum(axis=1)
    cons_df = cons_df[indi_cons.astype(bool)]

    rips_df = add_speed_events(rips_df)
    cons_df = add_speed_events(cons_df)

    # koko
    dfs = []
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    for i_match, (ax, match) in enumerate(zip(axes, [1, 2])):
        for i_event, event_df in enumerate([rips_df, cons_df]):
            df_cosines_melted = calc_df_cosines(event_df, match)
            test_kw_and_bm(df_cosines_melted)
            calc_corr(df_cosines_melted)
            ax = axes[i_match, i_event]
            ax, df = plot(ax, df_cosines_melted)
            event_str = ["SWR+", "SWR-"][i_event]
            df["match"] = match
            df["event"] = event_str
            dfs.append(df)
    fig.supylabel("Cosine of directions of SWRs")
    mngs.io.save(fig, "./tmp/figs/box/cos_of_SWR_directions.png")    
    plt.show()


    mm, iqr = mngs.gen.describe(
        df[(df.comparison == "ER")
       * (df.match == 1)
       * (df.event == "SWR+")
       * (df.set_size == 4)
       ].cosine
        , method="median"
        )
    np.degrees(np.arccos(mm)) # 133.2


    sns.boxplot(data=df,
                x="variable",
                y="value",
                )

    mngs.stats.brunner_munzel_test(
        df[df.variable=="EM_4"]["value"].astype(float),
        df[df.variable=="EM_8"]["value"].astype(float),
        )

    out_cosine["1_SWR+_EM_4.0"]

    df = pd.concat(dfs, axis=0)
    df["value"] = df["value"].astype(float)    
    df = df.rename(columns={"variable": "comparison", "value": "cosine"})
    df["set_size"] = np.nan
    set_sizes = np.zeros(len(df))
    set_sizes[mngs.gen.search("4", df["comparison"], as_bool=True)[0]] = 4
    set_sizes[mngs.gen.search("6", df["comparison"], as_bool=True)[0]] = 6
    set_sizes[mngs.gen.search("8", df["comparison"], as_bool=True)[0]] = 8

    df["set_size"] = set_sizes
    phase_combis = [cl[:2] for cl in df["comparison"]]
    df["comparison"] = phase_combis
    df[["event", "comparison", "set_size", "match"]].drop_duplicates()
    count = 0
    out = {}
    for match in df.match.unique():
        for event in df.event.unique():
            for comparison in ["FE", "FM", "FR", "EM", "ER", "MR"]:
                for set_size in df.set_size.unique():
                    print(match, event, comparison, set_size)
                    string = f"{match}_{event}_{comparison}_{set_size}"
                    out[string] = df[
                        (df.match == match)
                        * (df.event == event)
                        * (df.comparison == comparison)
                        * (df.set_size == set_size)
                    ]["cosine"]
                    count += 1
    out_cosine = mngs.gen.force_dataframe(out)
    out_cosine[out_cosine == ""] = np.nan
    mngs.io.save(out_cosine, "./tmp/figs/box/cos_of_SWR_directions.csv")
