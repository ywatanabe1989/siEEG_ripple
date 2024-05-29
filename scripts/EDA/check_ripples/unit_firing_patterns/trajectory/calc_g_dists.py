#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-04-29 07:53:47 (ywatanabe)"

from elephant.gpfa import GPFA
import quantities as pq
import mngs
import neo
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

# Functions
def get_subject_session_roi(lpath):
    subject = re.findall("Sub_[\w]{2}", lpath)[0][-2:]
    session = re.findall("Session_[\w]{2}", lpath)[0][-2:]
    roi = (
        re.findall("spike_times_[\w]{2,3}.pkl", lpath)[0]
        .split("spike_times_")[1]
        .split(".pkl")[0]
    )
    return subject, session, roi


def calc_g_dist(match, set_size):

    # Loads
    LPATHs = natsorted(glob("./data/Sub_*/Session_*/spike_times_*.pkl"))

    dfs = []
    info_df = pd.DataFrame()

    for lpath in LPATHs:
        subject, session, roi = get_subject_session_roi(lpath)

        try:
            if int(session) <= 2:
                trajs = mngs.io.load(
                    f"./data/Sub_{subject}/Session_{session}/traj_z_by_session_{roi}.npy"
                )
                trials_df = mngs.io.load(
                    f"./data/Sub_{subject}/Session_{session}/trials_info.csv"
                )

                # drops unnecessary rows
                # by match
                if match is not None:
                    trajs = trajs[trials_df.match == match]
                    trials_df = trials_df[trials_df.match == match]

                # # by correct
                # if False:
                #     trajs = trajs[trials_df.correct == 1]
                #     trials_df = trials_df[trials_df.correct == 1]

                # by set_size
                if set_size is not None:
                    trajs = trajs[trials_df.set_size == set_size]
                    trials_df = trials_df[trials_df.set_size == set_size]

                # PHASES, starts, ends, colors = define_phase_time()

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    g_phases = {}
                    for phase in PHASES:
                        g_phases[phase] = \
                            np.nanmedian(
                                trajs[:,:,GS_BINS_DICT[phase][0]:GS_BINS_DICT[phase][1]]
                                ,axis=-1)

                # # override g_phases["Retrieval"] for response_time
                # rt_bins = (
                #     (trials_df.response_time + 6) / (bin_size.rescale("s").magnitude)
                # ).astype(int)
                # g_retrieval = []
                # for i_rt, rt_bin in enumerate(rt_bins):
                #     with warnings.catch_warnings():
                #         warnings.simplefilter("ignore", RuntimeWarning)
                #         gr = np.nanmedian(
                #             trajs[
                #                 i_rt,
                #                 :,
                #                 rt_bin
                #                 - int(width_bins / 2) : rt_bin
                #                 + int(width_bins / 2),
                #             ],
                #             axis=-1,
                #         )
                #     g_retrieval.append(gr)
                # g_phases["Retrieval"] = np.vstack(g_retrieval)

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
                ) # repeat info_df_new
                info_df = pd.concat(
                    [
                        info_df,
                        info_df_new,
                    ],
                    axis=1,
                )
        except Exception as e:
            # print(e)
            pass

    return dfs, info_df


if __name__ == "__main__":
    from glob import glob
    import re
    from natsort import natsorted
    import sys
    sys.path.append(".")
    import utils

    (
        PHASES,
        PHASES_BINS_DICT,
        GS_BINS_DICT,
        COLORS_DICT,
        BIN_SIZE,
    ) = utils.define_phase_time()

    for match in [None, 1, 2]:    
        for set_size in [None, 4, 6, 8]:
            """
            match = 1
            set_size = 4
            """                
            dfs, info_df = calc_g_dist(match, set_size)

            try:

                # indices
                indi_hipp = mngs.general.search(
                    ["AHL", "AHR", "PHL", "PHR"], info_df.T.ROI, as_bool=True
                )[0]
                indi_ec = mngs.general.search(
                    ["ECL", "ECR"], info_df.T.ROI, as_bool=True
                )[0]
                indi_amy = mngs.general.search(
                    ["AL", "AR"], info_df.T.ROI, as_bool=True
                )[0]
            except Exception as e:
                print(e)
                import ipdb; ipdb.set_trace()

            for roi_str, indi_roi in zip(
                ["Hipp.", "EC", "Amy."], [indi_hipp, indi_ec, indi_amy]
            ):
                g_phases = pd.concat(
                    [
                        mngs.general.force_dataframe(dfs[ii])
                        for ii in range(len(dfs))
                    ]
                )[indi_roi]

                nan_indi = (
                    np.stack(
                        [np.isnan(np.vstack(g_phases[phase])) for phase in PHASES]
                    )
                    .any(axis=0)
                    .any(axis=-1)
                )
                g_phases = g_phases[~nan_indi]

                # df_mm = pd.DataFrame(index=PHASES, columns=PHASES)
                # df_ss = pd.DataFrame(index=PHASES, columns=PHASES)
                # for p1, p2 in product(PHASES, PHASES):
                #     dists = norm(
                #         np.stack(g_phases[p1] - g_phases[p2]), axis=-1
                #     ).round(3)
                #     mm, iqr = mngs.gen.describe(dists, method="median")

                #     df_mm.loc[p1, p2] = mm.round(3)
                #     df_ss.loc[p1, p2] = iqr.round(3)

                # out_df = pd.concat([df_mm, df_ss], axis=1)
                # spath = (
                #     f"./tmp/g_dist_z_{z_by}/{roi_str}_{set_size}_match_{match}.csv"
                # )
                # mngs.io.save(
                #     out_df,
                #     spath,
                # )

                # raw data for hist plot
                data = {}
                for p1, p2 in combinations(PHASES, 2):
                    data[f"{p1[0]}{p2[0]}"] = norm(
                        np.stack(g_phases[p1] - g_phases[p2]).astype(float),
                        axis=-1,
                    )
                out_df_2 = mngs.general.force_dataframe(data)
                spath = f"./res/figs/box/g_dist_z_by_session/{roi_str}_{set_size}_match_{match}_raw_dists.csv"
                mngs.io.save(
                    out_df_2,
                    spath,
                )

                # # stats
                # count = 0
                # pval = 1
                # is_significant = False
                # for ii, (p1, p2) in enumerate(combinations(PHASES, 2)):
                #     comp_1_str = f"{p1[0]}-{p2[0]}"
                #     for jj, (p3, p4) in enumerate(combinations(PHASES, 2)):
                #         comp_2_str = f"{p3[0]}-{p4[0]}"
                #         if ii < jj:
                #             count += 1
                #             stats, pval = brunnermunzel(
                #                 norm(
                #                     np.stack(g_phases[p1] - g_phases[p2]).astype(
                #                         float
                #                     ),
                #                     axis=-1,
                #                 ),
                #                 norm(
                #                     np.stack(g_phases[p3] - g_phases[p4]).astype(
                #                         float
                #                     ),
                #                     axis=-1,
                #                 ),
                #             )
                #             pval *= 15
                #             if pval < 0.05: # .1
                #                 # if True:
                #                 print(
                #                     f"{p1[0]}-{p2[0]}, {p3[0]}-{p4[0]}",
                #                     pval.round(3),
                #                 )
                #                 is_significant = True
                # if is_significant:
                #     pass


"""
Saved to: ./tmp/g_dist_z_by_trial/Hipp._None_match_None.csv

F-M, F-R 0.002
F-R, E-M 0.0
E-M, E-R 0.001

Saved to: ./tmp/g_dist_z_by_trial/Amy._None_match_None.csv

F-R, E-M 0.001

Saved to: ./tmp/g_dist_z_by_trial/Hipp._4_match_None.csv

E-M, E-R 0.001

Saved to: ./tmp/g_dist_z_by_trial/Hipp._None_match_2.csv

F-M, F-R 0.003
F-R, E-M 0.0
"""
# z_by = "by_session"
# roi_str = "Hipp."
# match = 1
# set_size = 4
# for set_size in [4, 6, 8]:
#     data = mngs.io.load(
#         f"./tmp/g_dist_z_{z_by}/{roi_str}_{set_size}_match_{match}_raw_dists.csv"
#     )
#     print(describe(np.log10(data["ER"]), "median"))
