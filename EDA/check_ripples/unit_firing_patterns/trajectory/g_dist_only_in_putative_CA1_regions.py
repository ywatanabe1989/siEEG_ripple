#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-03-16 17:33:13 (ywatanabe)"

import mngs
import numpy as np
import pandas as pd
from numpy.linalg import norm
import mngs
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import seaborn as sns
import random
from scipy.stats import brunnermunzel
from itertools import product, combinations
import warnings
from itertools import combinations
from scipy.stats import kruskal

# Functions
def calc_g_dist(z_by, match, set_size, replace_SWR_periods_with_median=False):
    # # Loads
    # LPATHs = natsorted(glob("./data/Sub_*/Session_*/spike_times_*.pkl"))

    dfs = []
    info_df = pd.DataFrame()

    for subject, roi in ROIs.items():
        subject = f"{subject:02d}"

        for session in ["01", "02"]:
            if replace_SWR_periods_with_median:
                trajs = mngs.io.load(
                    f"./data/Sub_{subject}/Session_{session}/"
                    f"traj_z_{z_by}_{roi}_replaced_SWR_periods_with_median.npy"
                )
            else:
                trajs = mngs.io.load(
                    f"./data/Sub_{subject}/Session_{session}/"
                    f"traj_z_{z_by}_{roi}.npy"
                )
                
            trials_df = mngs.io.load(
                f"./data/Sub_{subject}/Session_{session}/trials_info.csv"
            )

            # drops unnecessary rows
            # by match
            if match is not None:
                trajs = trajs[trials_df.match == match]
                trials_df = trials_df[trials_df.match == match]

            # by correct
            if False:
                trajs = trajs[trials_df.correct == 1]
                trials_df = trials_df[trials_df.correct == 1]

            # by set_size
            if set_size is not None:
                trajs = trajs[trials_df.set_size == set_size]
                trials_df = trials_df[trials_df.set_size == set_size]

            g_phases = {}
            for phase in PHASES:
                g_phases[phase] = np.nanmedian(
                    trajs[:, :, GS_BINS_DICT[phase][0] : GS_BINS_DICT[phase][1]],
                    axis=-1,
                )

            dfs.append(g_phases)

            info_df_new = pd.DataFrame(
                pd.Series(
                    {
                        "subject": subject,
                        "session": session,
                        "ROI": roi,
                        "set_size": set_size,
                    }
                )
            )
            info_df_new = pd.concat(
                [info_df_new for _ in range(len(g_phases["Fixation"]))], axis=1
            )
            info_df = pd.concat(
                [
                    info_df,
                    info_df_new,
                ],
                axis=1,
            )

    return dfs, info_df


def extract_g_phases(dfs, info_df):
    # g_phases
    g_phases = pd.concat(
        [mngs.general.force_dataframe(dfs[ii]) for ii in range(len(dfs))]
    )
    nan_indi = (
        np.stack([np.isnan(np.vstack(g_phases[phase])) for phase in PHASES])
        .any(axis=0)
        .any(axis=-1)
    )
    g_phases = g_phases[~nan_indi]
    return g_phases


def calc_dist_between_gs(g_phases):
    # raw data for hist plot
    data = {}
    for p1, p2 in combinations(PHASES, 2):
        data[f"{p1[0]}{p2[0]}"] = norm(
            np.stack(g_phases[p1] - g_phases[p2]).astype(float),
            axis=-1,
        )
    out_df = mngs.general.force_dataframe(data)
    spath = (
        f"./tmp/figs/box/g_dist_z_{z_by}_replaced_SWR_periods_with_median/"
        f"{set_size}_match_{match}_raw_dists.csv"
    )
    mngs.io.save(
        out_df,
        spath,
    )
    return out_df


def test_bm(g_phases):
    # stats
    count = 0
    pval = 1
    is_significant = False
    for ii, (p1, p2) in enumerate(combinations(PHASES, 2)):
        comp_1_str = f"{p1[0]}-{p2[0]}"
        for jj, (p3, p4) in enumerate(combinations(PHASES, 2)):
            comp_2_str = f"{p3[0]}-{p4[0]}"
            if ii < jj:
                count += 1
                stats, pval = brunnermunzel(
                    norm(
                        np.stack(g_phases[p1] - g_phases[p2]).astype(float),
                        axis=-1,
                    ),
                    norm(
                        np.stack(g_phases[p3] - g_phases[p4]).astype(float),
                        axis=-1,
                    ),
                )
                pval *= 15
                if pval < 0.1:
                    # if True:
                    print(
                        f"{p1[0]}-{p2[0]}, {p3[0]}-{p4[0]}",
                        pval.round(3),
                    )
                    is_significant = True
    if is_significant:
        pass


if __name__ == "__main__":
    from glob import glob
    import re
    from natsort import natsorted
    import sys

    sys.path.append(".")
    import utils

    # Parameters
    (
        PHASES,
        PHASES_BINS_DICT,
        GS_BINS_DICT,
        COLORS_DICT,
        BIN_SIZE,
    ) = utils.define_phase_time()

    ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")
    z_by = "by_session"
    # MTL_region = "Hipp."
    # match = 1
    # set_size = 4
    # for set_size in [4, 6, 8]:
    #     data = mngs.io.load(
    #         f"./tmp/g_dist_z_{z_by}/{MTL_region}_{set_size}_match_{match}_raw_dists.csv"
    #     )
    #     print(describe(np.log10(data["ER"]), "median"))

    for match in [None, 1, 2]:
        for set_size in [None, 4, 6, 8]:
            """
            match = 1
            set_size = None
            """
            dfs, info_df = calc_g_dist(z_by, match, set_size)

            g_phases = extract_g_phases(dfs, info_df)
            dists_between_gs = calc_dist_between_gs(g_phases)
            test_bm(g_phases)

    match = 1
    data = []
    for set_size in [4, 6, 8]:
        lpath = (
            f"./tmp/figs/box/g_dist_z_{z_by}_replaced_SWR_periods_with_median/"
            f"{set_size}_match_{match}_raw_dists.csv"
        )
        data.append(mngs.io.load(lpath)["ER"])

    kruskal(*data)
    
    for d1, d2 in combinations(data, 2):
        w, p, d, e = mngs.stats.brunner_munzel_test(d1, d2)
        print(mngs.gen.describe(d1, method="median"))
        print(mngs.gen.describe(d2, method="median"))
        print(p)

        
    plt.boxplot(data, positions=[4,6,8])
    plt.show()
