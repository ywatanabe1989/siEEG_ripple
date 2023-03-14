#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-03-05 13:59:26 (ywatanabe)"

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-02-27 08:43:33 (ywatanabe)"

"""
./tmp/figs/time_dependent_dist
./tmp/figs/hist/traj_dist
./tmp/figs/hist/traj_pos
"""

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# from elephant.gpfa import GPFA
import mngs

# import neo
import numpy as np
import pandas as pd

# import ffmpeg
# from matplotlib import animation
import os
from bisect import bisect_right
import seaborn as sns
import random
import sys

sys.path.append(".")
import utils
from itertools import product, combinations
from natsort import natsorted

# Functions
def rr_distance(coords_1, coords_2):  # round-robin
    dists = []
    for i1 in range(len(coords_1)):
        for i2 in range(len(coords_2)):
            dist = mngs.linalg.nannorm(coords_1[i1] - coords_2[i2])
            dists.append(dist)
    return dists


def summarize_to_df_dists(
    events_df, subject, session, set_size, phase_1, phase_2, delta_bin
):
    events_df_session = events_df[
        (events_df.subject == subject)
        * (events_df.session == session)
        * (events_df.set_size == set_size)
        # * (events_df.match == 1)
    ]
    # delta_bin = 2
    pre_1 = events_df_session[f"-{delta_bin}"][events_df_session.phase == phase_1]
    post_1 = events_df_session[f"{delta_bin}"][events_df_session.phase == phase_1]
    pre_2 = events_df_session[f"-{delta_bin}"][events_df_session.phase == phase_2]
    post_2 = events_df_session[f"{delta_bin}"][events_df_session.phase == phase_2]

    dic_coords = {
        f"pre_{phase_1[0]}": pre_1,
        f"post_{phase_1[0]}": post_1,
        f"pre_{phase_2[0]}": pre_2,
        f"post_{phase_2[0]}": post_2,
    }

    dic_dists = {}
    for (a_key, a_val), (b_key, b_val) in combinations(dic_coords.items(), 2):
        # for (a_key, a_val), (b_key, b_val) in product(dic_coords.items(), dic_coords.items()): # fixme
        key = f"{a_key}-{b_key}"
        dists = rr_distance(np.array(a_val), np.array(b_val))
        dic_dists[key] = dists

    df_dists = mngs.gen.force_dataframe(dic_dists).replace({"": np.nan})
    df_dists = df_dists.melt()
    df_dists = df_dists[~df_dists.value.isna()]
    df_dists["set_size"] = set_size
    df_dists = df_dists.rename(columns={"variable": "comparison", "value": "distance"})
    return df_dists


def calc_dists_df(event_df, phase_1, phase_2, delta_bin):
    df_dists = []
    for subject in ROIs.keys():
        subject = f"{int(subject):02d}"
        for session in ["01", "02"]:
            for set_size in [4, 6, 8]:
                _df_dists = summarize_to_df_dists(
                    event_df, subject, session, set_size, phase_1, phase_2, delta_bin
                )
                df_dists.append(_df_dists)

    df_dists = pd.concat(df_dists)
    return df_dists


def sort_df_dists(df_dists):
    out = {}
    for comp in df_dists.comparison.unique():
        for set_size in df_dists.set_size.unique():
            dist = df_dists[
                (df_dists.comparison == comp) * (df_dists.set_size == set_size)
            ]["distance"]
            out[f"{comp}_{set_size}"] = dist
    return mngs.gen.force_dataframe(out)


def plot_df_dists(df_dists_all, phase_1, phase_2):
    order = [
        # f"pre_{phase_1[0]}-pre_{phase_1[0]}",
        # f"post_{phase_1[0]}-post_{phase_1[0]}",
        # f"pre_{phase_2[0]}-pre_{phase_2[0]}",
        # f"post_{phase_2[0]}-post_{phase_2[0]}",
        f"pre_{phase_1[0]}-post_{phase_1[0]}",
        f"pre_{phase_1[0]}-pre_{phase_2[0]}",
        f"pre_{phase_1[0]}-post_{phase_2[0]}",
        f"post_{phase_1[0]}-pre_{phase_2[0]}",
        f"post_{phase_1[0]}-post_{phase_2[0]}",
        f"pre_{phase_2[0]}-post_{phase_2[0]}",
    ]
    df_dists_all["set_size-event"] = mngs.ml.utils.merge_labels(
        df_dists_all["set_size"], df_dists_all["event"]
    )
    hue_order = natsorted(df_dists_all["set_size-event"].unique())
    fig, ax = plt.subplots()
    sns.boxplot(
        data=df_dists_all,
        x="comparison",
        order=order,
        y="distance",
        # hue="event",
        # hue_order=["SWR+", "SWR-_across", "SWR-_within"],
        hue="set_size-event",
        hue_order=hue_order,
        ax=ax,
        showfliers=False,
    )
    # sns.stripplot(
    #     data=df_dists_all,
    #     x="comparison",
    #     order=order,
    #     y="distance",
    #     hue="set_size-event",
    #     hue_order=hue_order,
    #     ax=ax,
    # )
    return fig


def bm_test(df_dists):
    for comp in df_dists.comparison.unique():
        for ss1, ss2 in combinations([4, 6, 8], 2):
            d1 = df_dists[(df_dists.comparison == comp) * (df_dists.set_size == ss1)]
            d2 = df_dists[(df_dists.comparison == comp) * (df_dists.set_size == ss2)]
            w, p, d, e = mngs.stats.brunner_munzel_test(np.log10(d1.distance+1e-5), np.log10(d2.distance+1e-5))
            if p * 3 < 0.05:
                print(comp, ss1, ss2, round(p * 3, 3))
        print()


def calc_corr(df_dists):
    mngs.gen.fix_seeds(42, np=np)
    for comp in df_dists.comparison.unique():
        df_comp = df_dists[df_dists.comparison == comp].copy()
        df_comp.distance = np.log10(df_comp.distance+1e-5)
        corr_obs = np.corrcoef(df_comp.distance, df_comp.set_size)[0, 1]
        corr_shuffled = [
            np.corrcoef(df_comp.distance, np.random.permutation(df_comp.set_size))[0, 1]
            for _ in range(1000)
        ]
        rank = bisect_right(corr_shuffled, corr_obs)
        print(comp)
        print(f"corr_obs: {round(corr_obs, 2)}")
        print(f"rank: {rank}")
        print()


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
    cons_within_df = utils.rips.add_coordinates(
        utils.rips.load_rips_df_with_traj(BIN_SIZE, is_control=True)
    )
    cons_across_df = utils.rips.add_coordinates(
        utils.rips.load_rips_df_with_traj(BIN_SIZE, is_within_control=True)
    )

    rips_df_orig = rips_df.copy()
    cons_within_df_orig = cons_within_df
    cons_across_df_orig = cons_across_df
    for match in [1, 2]:
        rips_df = rips_df_orig[rips_df_orig.match == match]
        cons_within_df = cons_within_df_orig[cons_within_df_orig.match == match]
        cons_across_df = cons_across_df_orig[cons_across_df_orig.match == match]

        rips_df["-0"] = rips_df["0"]
        cons_within_df["-0"] = cons_within_df["0"]
        cons_across_df["-0"] = cons_across_df["0"]

        phase_1 = "Encoding"
        phase_2 = "Retrieval"

        delta_bin = 1

        df_dists_rips = calc_dists_df(rips_df, phase_1, phase_2, delta_bin)
        df_dists_cons_within = calc_dists_df(
            cons_within_df, phase_1, phase_2, delta_bin
        )
        df_dists_cons_across = calc_dists_df(
            cons_across_df, phase_1, phase_2, delta_bin
        )
        # df_dists_rips["distance"] = np.log10(df_dists_rips["distance"]+1e-5)
        # df_dists_cons_within["distance"] = np.log10(df_dists_cons_within["distance"]+1e-5)
        # df_dists_cons_across["distance"] = np.log10(df_dists_cons_across["distance"]+1e-5)
        mngs.gen.describe(df_dists_rips[(df_dists_rips.comparison == "post_E-post_R")
                                        * (df_dists_rips.set_size == 4)].distance,
                          method="median")


        df_dists_rips["event"] = "SWR+"
        df_dists_cons_within["event"] = "SWR-_within"
        df_dists_cons_across["event"] = "SWR-_across"
        df_dists_all = pd.concat(
            [df_dists_rips, df_dists_cons_within, df_dists_cons_across]
        )

        mngs.io.save(
            sort_df_dists(df_dists_rips),
            f"./tmp/figs/box/dist_between_pre_and_post_SWR/match_{match}/SWR+.csv",
        )
        mngs.io.save(
            sort_df_dists(df_dists_cons_within),
            f"./tmp/figs/box/dist_between_pre_and_post_SWR/match_{match}/SWR-_within.csv",
        )
        mngs.io.save(
            sort_df_dists(df_dists_cons_across),
            f"./tmp/figs/box/dist_between_pre_and_post_SWR/match_{match}/SWR-_across.csv",
        )

        # fig_all = plot_df_dists(df_dists_all, phase_1, phase_2)
        # plt.show()
        fig_rips = plot_df_dists(df_dists_rips, phase_1, phase_2)
        plt.show()
        # df_dists_rips[["comparison", "distance"]].pivot_table(columns="comparison", aggfunc="mean")

        fig_cons_within = plot_df_dists(df_dists_cons_within, phase_1, phase_2)
        fig_cons_across = plot_df_dists(df_dists_cons_across, phase_1, phase_2)

        mngs.io.save(
            fig_rips,
            f"./tmp/figs/box/dist_between_pre_and_post_SWR/match_{match}/SWR+.png",
        )
        mngs.io.save(
            fig_cons_within,
            f"./tmp/figs/box/dist_between_pre_and_post_SWR/match_{match}/SWR-_within.png",
        )
        mngs.io.save(
            fig_cons_across,
            f"./tmp/figs/box/dist_between_pre_and_post_SWR/match_{match}/SWR-_across.png",
        )

        bm_test(df_dists_rips)
        bm_test(df_dists_cons_across)        
        bm_test(df_dists_cons_within)

        calc_corr(df_dists_rips)
        calc_corr(df_dists_cons_across)        
        calc_corr(df_dists_cons_within)

