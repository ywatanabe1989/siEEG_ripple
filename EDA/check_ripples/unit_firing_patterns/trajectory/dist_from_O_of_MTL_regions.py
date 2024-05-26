#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-05-25 08:46:05 (ywatanabe)"

import matplotlib
import mngs

matplotlib.use("TkAgg")
import random
from itertools import combinations
from pprint import pprint

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.linalg import norm

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def get_rois(MTL_region):
    if MTL_region == "Hipp.":
        return ["AHL", "AHR", "PHL", "PHR"]
    if MTL_region == "EC":
        return ["ECL", "ECR"]
    if MTL_region == "Amy.":
        return ["AL", "AR"]


def collect_traj(MTL_region):
    subjects = ["01", "02", "03", "04", "05", "06", "07", "08", "09"]
    trajs = []
    for subject in subjects:
        for session in ["01", "02"]:
            for roi in get_rois(MTL_region):
                try:
                    traj = mngs.io.load(
                        f"./data/Sub_{subject}/Session_{session}/traj_z_by_session_{roi}.npy"
                    )
                    trajs.append(traj)
                except Exception as e:
                    print(e)
    return np.vstack(trajs)


def calc_dist(traj_MTL_region):
    norm_nonnan_MTL_region = {}
    for i_bin in range(traj_MTL_region.shape[-1]):  # i_bin = 0
        traj_MTL_region_i_bin = traj_MTL_region[..., i_bin]
        norm_MTL_region_i_bin = norm(
            traj_MTL_region_i_bin[
                ~np.isnan(traj_MTL_region_i_bin).any(axis=1)
            ],
            axis=-1,
        )
        norm_nonnan_MTL_region[i_bin] = norm_MTL_region_i_bin
    return mngs.gen.force_dataframe(norm_nonnan_MTL_region)


def calc_mean_and_ci(dist):
    dist_mm = np.nanmean(dist, axis=0)
    dist_sd = np.nanstd(dist, axis=0)
    dist_nn = (~np.isnan(dist)).astype(int).sum(axis=0)
    dist_ci = 1.96 * dist_mm / (dist_sd * dist_nn)
    return dist_mm, dist_ci


def calc_dist_between_gs(traj):
    trial_gs = {
        pp: np.nanmedian(traj[:, :, ss:ee], axis=-1)
        for pp, (ss, ee) in GS_BINS_DICT.items()
    }

    dist_between_gs = {}
    for p1, p2 in list(combinations(GS_BINS_DICT.keys(), 2)):
        # print(f"{p1[0]}{p2[0]}")
        dist_between_gs[f"{p1[0]}{p2[0]}"] = [
            mngs.linalg.nannorm(v) for v in trial_gs[p1] - trial_gs[p2]
        ]
    return dist_between_gs


if __name__ == "__main__":
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

    traj_Hipp = collect_traj("Hipp.")
    traj_EC = collect_traj("EC")
    traj_Amy = collect_traj("Amy.")

    dist_Hipp = calc_dist(traj_Hipp)
    dist_EC = calc_dist(traj_EC)
    dist_Amy = calc_dist(traj_Amy)

    # Distance from O
    # for line plot
    dist_Hipp_mm, dist_Hipp_ci = calc_mean_and_ci(dist_Hipp)
    dist_EC_mm, dist_EC_ci = calc_mean_and_ci(dist_EC)
    dist_Amy_mm, dist_Amy_ci = calc_mean_and_ci(dist_Amy)

    df_mean_and_ci = mngs.gen.force_dataframe(
        {
            "Hipp._under": dist_Hipp_mm - dist_Hipp_ci,
            "Hipp._mean": dist_Hipp_mm,
            "Hipp._upper": dist_Hipp_mm + dist_Hipp_ci,
            "EC_under": dist_EC_mm - dist_EC_ci,
            "EC_mean": dist_EC_mm,
            "EC_upper": dist_EC_mm + dist_EC_ci,
            "Amy._under": dist_Amy_mm - dist_Amy_ci,
            "Amy._mean": dist_Amy_mm,
            "Amy._upper": dist_Amy_mm + dist_Amy_ci,
        }
    )
    mngs.io.save(df_mean_and_ci, "./res/figs/line/MTL_regions/dist_from_O.csv")

    # for boxplot
    dist_from_O_Hipp_all = np.array(dist_Hipp).reshape(-1)
    dist_from_O_EC_all = np.array(dist_EC).reshape(-1)
    dist_from_O_Amy_all = np.array(dist_Amy).reshape(-1)
    df_dist_all = mngs.gen.force_dataframe(
        {
            "Hipp.": dist_from_O_Hipp_all,
            "EC": dist_from_O_EC_all,
            "Amy.": dist_from_O_Amy_all,
        }
    )
    mngs.io.save(df_dist_all, "./res/figs/box/MTL_regions/dist_from_O_all.csv")

    # Distance between gs
    dist_between_gs_Hipp_all = np.hstack(
        [val for val in calc_dist_between_gs(traj_Hipp).values()]
    )
    dist_between_gs_EC_all = np.hstack(
        [val for val in calc_dist_between_gs(traj_EC).values()]
    )
    dist_between_gs_Amy_all = np.hstack(
        [val for val in calc_dist_between_gs(traj_Amy).values()]
    )
    df_dist_between_gs_all = mngs.gen.force_dataframe(
        {
            "Hipp.": dist_between_gs_Hipp_all,
            "EC": dist_between_gs_EC_all,
            "Amy.": dist_between_gs_Amy_all,
        }
    )
    mngs.io.save(
        df_dist_between_gs_all,
        "./res/figs/box/MTL_regions/dist_between_gs_all.csv",
    )
    # mngs.io.save(df_dist_all, "./res/figs/box/MTL_regions/dist_from_O_all.csv")

    mngs.io.save(
        pd.DataFrame(dist_between_gs_Hipp),
        "./res/figs/box/MTL_regions/dist_between_gs_Hipp.csv",
    )
    mngs.io.save(
        pd.DataFrame(dist_between_gs_EC),
        "./res/figs/box/MTL_regions/dist_between_gs_EC.csv",
    )
    mngs.io.save(
        pd.DataFrame(dist_between_gs_Amy),
        "./res/figs/box/MTL_regions/dist_between_gs_Amy.csv",
    )

    # dist_Hipp_mm

    # print(mngs.gen.describe(dist_Hipp, method="mean"))
    # print(mngs.gen.describe(dist_EC, method="mean"))
    # print(mngs.gen.describe(dist_Amy, method="mean"))

    # print(mngs.gen.describe(dist_Hipp, method="median"))
    # print(mngs.gen.describe(dist_EC, method="median"))
    # print(mngs.gen.describe(dist_Amy, method="median"))

    # w, p, dof, eff = mngs.stats.brunner_munzel_test(
    #     np.hstack(np.array(dist_Hipp)),
    #     np.hstack(np.array(dist_EC)),
    # )
    # print(p)

    # w, p, dof, eff = mngs.stats.brunner_munzel_test(
    #     np.hstack(np.array(dist_EC)),
    #     np.hstack(np.array(dist_Amy)),
    # )
    # print(p)

    # w, p, dof, eff = mngs.stats.brunner_munzel_test(
    #     np.hstack(np.array(dist_Amy)),
    #     np.hstack(np.array(dist_Hipp)),
    # )
    # print(p)

    # fig, ax = plt.subplots()
    # ax.boxplot(
    #     np.hstack(np.array(dist_Hipp)),
    #     positions=[0],
    #     showfliers=False,
    # )
    # ax.boxplot(
    #     np.hstack(np.array(dist_EC)),
    #     positions=[1],
    #     showfliers=False,
    # )
    # ax.boxplot(
    #     np.hstack(np.array(dist_Amy)),
    #     positions=[2],
    #     showfliers=False,
    # )
    # ax.set_yscale("log")
    # plt.show()

    # df = mngs.gen.force_dataframe(
    #     {
    #         "Hipp.": np.log10(np.hstack(np.array(dist_Hipp))),
    #         "EC": np.log10(np.hstack(np.array(dist_EC))),
    #         "Amy": np.log10(np.hstack(np.array(dist_Amy))),
    #     }
    # )
    # mngs.io.save(df, "./tmp/figs/box/log10_dist_from_O_by_MTL_region.csv")

    # mm_Hipp, ss_Hipp = dist_Hipp.mean(axis=0), dist_Hipp.std(axis=0)
    # mm_EC, ss_EC = dist_EC.mean(axis=0), dist_EC.std(axis=0)
    # mm_Amy, ss_Amy = dist_Amy.mean(axis=0), dist_Amy.std(axis=0)

    # df = pd.DataFrame(
    #     {
    #         "under_Hipp.": mm_Hipp - ss_Hipp,
    #         "mean_Hipp.": mm_Hipp,
    #         "upper_Hipp.": mm_Hipp + ss_Hipp,
    #         "under_EC": mm_EC - ss_EC,
    #         "mean_EC": mm_EC,
    #         "upper_EC": mm_EC + ss_EC,
    #         "under_Amy.": mm_Amy - ss_Amy,
    #         "mean_Amy.": mm_Amy,
    #         "upper_Amy.": mm_Amy + ss_Amy,
    #     }
    # )
    # mngs.io.save(df, "./tmp/figs/line/repr_traj.csv")
