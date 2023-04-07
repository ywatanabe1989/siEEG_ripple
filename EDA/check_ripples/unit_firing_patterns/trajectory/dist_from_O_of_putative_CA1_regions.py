#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-03-22 21:49:33 (ywatanabe)"

import mngs
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from itertools import combinations
from scipy.linalg import norm
import pandas as pd
from pprint import pprint
import random


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
            traj_MTL_region_i_bin[~np.isnan(traj_MTL_region_i_bin).any(axis=1)], axis=-1
        )
        norm_nonnan_MTL_region[i_bin] = norm_MTL_region_i_bin
    return mngs.gen.force_dataframe(norm_nonnan_MTL_region)


if __name__ == "__main__":

    import mngs
    ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")

    # trajs = []
    dists_z = []
    for subject, roi in ROIs.items():
        subject = f"{subject:02d}"
        for session in ["01", "02"]:
                traj = mngs.io.load(
                    f"./data/Sub_{subject}/Session_{session}/traj_z_by_session_{roi}.npy"
                )
                dist = mngs.linalg.nannorm(traj, axis=1)
                dist_z = zscore(dist, axis=1)
                dists_z.append(dist_z)
                # trajs.append(traj)

    from scipy.stats import zscore
    # trajs = np.vstack(trajs)
    # dists = mngs.linalg.nannorm(trajs, axis=1)

    dists_z = np.vstack(dists_z)

    dists = dists_z

    mm = np.nanmean(dists, axis=0)
    sd = np.nanstd(dists, axis=0)
    nn = np.sum(~np.isnan(dists), axis=0)
    ci = 1.96 * sd / nn

    import scipy

    fig, ax = plt.subplots()
    xx = np.linspace(0,8,160)
    ax.plot(xx, mngs.dsp.gaussian_filter1d(mm, 10))
    ax.plot(xx, np.zeros(160))
    ax.axvline(x=1)
    ax.axvline(x=3)    
    ax.axvline(x=6)
    # plt.plot(xx, -1*np.ones(160))    
    plt.show()
            
    #         for roi in get_rois(MTL_region):
    #             try:
    #                 traj = mngs.io.load(
    #                     f"./data/Sub_{subject}/Session_{session}/traj_z_by_session_{roi}.npy"
    #                 )
    #                 trajs.append(traj)
    #             except Exception as e:
    #                 print(e)
    # return np.vstack(trajs)
    
    # traj_Hipp = collect_traj("Hipp.")
    # traj_EC = collect_traj("EC")
    # traj_Amy = collect_traj("Amy.")


    # dist_Hipp = calc_dist(traj_Hipp)
    # dist_EC = calc_dist(traj_EC)
    # dist_Amy = calc_dist(traj_Amy)

    # mngs.gen.describe(dist_Hipp, method="mean")
    # mngs.gen.describe(dist_EC, method="mean")
    # mngs.gen.describe(dist_Amy, method="mean")    

    # mngs.gen.describe(dist_Hipp, method="median")
    # mngs.gen.describe(dist_EC, method="median")
    # mngs.gen.describe(dist_Amy, method="median")    
    
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

    # import matplotlib

    # matplotlib.use("TkAgg")
    # import matplotlib.pyplot as plt

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


    # # mngs.linalg.nannorm(, axis=1)
    # # traj_Hipp.shape
    # # np.isnan(traj_Hipp)
    # # (~np.isnan(traj_Hipp).any(axis=-1).any(axis=-1)).sum()

    # # norm_Hipp = norm(traj_Hipp, axis=1)
    # # norm_EC = norm(traj_EC, axis=1)
    # # norm_Amy = norm(traj_Amy, axis=1)

    # # mm_Hipp, ss_Hipp = norm_Hipp.mean(axis=0), norm_Hipp.std(axis=0)
    # # mm_EC, ss_EC = norm_EC.mean(axis=0), norm_EC.std(axis=0)
    # # mm_Amy, ss_Amy = norm_Amy.mean(axis=0), norm_Amy.std(axis=0)

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
    # # plt.plot(norm(np.median(traj_Hipp, axis=0), axis=0))


    # # subject = "06"  # f"{int(list(ROIs.keys())[2]):02d}"  # 04
    # # session = "02"  # 01
    # # # subject = "06"
    # # # session = "02"
    # # # i_trial = 5 + 1
    # # i_trial = 4#random.randint(0,49)
    # # # for i_trial in range(49):

    # # traj_Hipp = mngs.io.load(
    # #     f"./data/Sub_{subject}/Session_{session}/traj_z_by_session_Hipp.npy"
    # # )
    # # traj_ECL = mngs.io.load(
    # #     f"./data/Sub_{subject}/Session_{session}/traj_z_by_session_ECL.npy"
    # # )
    # # traj_AL = mngs.io.load(
    # #     f"./data/Sub_{subject}/Session_{session}/traj_z_by_session_AL.npy"
    # # )


    # # # plt.plot(traj_Hipp[5].T)
    # # # plt.show()

    # # trajs = dict(
    # #     Hipp=traj_Hipp,
    # #     ECL=traj_ECL,
    # #     AL=traj_AL,
    # # )

    # # for traj_str, traj in trajs.items():
    # #     # trajs[traj_str] = trajs[traj_str][i_trial]
    # #     trajs[traj_str] = np.array(
    # #         [
    # #             scipy.ndimage.gaussian_filter1d(traj[:,i_dim], sigma=0.01, axis=-1)
    # #             for i_dim in range(3)]
    # #     ).transpose(1,0,2)

    # # # np.nanmedian(traj
    # # # scipy.ndimage.gaussian_filter1d(    , sigma=3)
    # # PHASES = ["Fixation", "Encoding", "Maintenance", "Retrieval"]
    # # starts = [0, 20, 60, 120]
    # # ends = [20, 60, 120, 160]
    # # centers = ((np.array(starts) + np.array(ends)) / 2).astype(int)
    # # width = 10
    # # colors = ["gray", "blue", "green", "red"]
    # # linestyles = ["solid", "dotted", "dashed"]
    # # xx = np.linspace(0, 8, 160) -6 # (np.arange(160)+1) / 160 / (50*1e-3)

    # # out_dict = {}
    # # fig, axes = plt.subplots(nrows=len(trajs), sharex=True, sharey=True)
    # # for i_ax, (ax, (traj_str, traj)) in enumerate(zip(axes, trajs.items())):
    # #     for i_dim in range(traj.shape[1]):
    # #         for ss, ee, cc, pp in zip(starts, ends, colors, PHASES):
    # #             yy = traj[i_trial][i_dim]
    # #             ax.plot(
    # #                 xx[ss:ee],
    # #                 yy[ss:ee],
    # #                 label=f"Factor {i_dim+1}",
    # #                 # color=colors[i_dim],
    # #                 color=cc,
    # #                 linestyle=linestyles[i_dim],
    # #             )
    # #             out_dict[f"xx_{traj_str}_{pp}_Factor_{i_dim}"] = xx[ss:ee]
    # #             out_dict[f"yy_{traj_str}_{pp}_Factor_{i_dim}"] = yy[ss:ee]
    # #     ax.set_ylabel(traj_str)
    # #     # ax.legend(loc="upper right")
    # # mngs.io.save(fig, "./tmp/figs/line/representative_trajectory/fig.png")
    # # plt.show()
    # # out_df = mngs.general.force_dataframe(out_dict)
    # # mngs.io.save(out_df, "./tmp/figs/line/representative_trajectory/data.csv")


    # # gs = {}
    # # for traj_str, traj in trajs.items():
    # #     for cc, pp in zip(centers, PHASES):
    # #         g = np.median(traj[i_trial, :, int(cc - width / 2) : int(cc + width / 2)], axis=-1)
    # #         gs[f"{traj_str}_{pp}"] = g
    # # mngs.io.save(pd.DataFrame(gs), "./tmp/figs/line/representative_trajectory/gs.csv")


    # # dfs = {}
    # # for roi in ["Hipp", "ECL", "AL"]:
    # #     df = pd.DataFrame(columns=PHASES, index=PHASES)
    # #     for ii, jj in combinations(np.arange(4), 2):
    # #         df.iloc[ii, jj] = norm(gs[roi][ii] - gs[roi][jj])
    # #     dfs[roi] = df


    # # pprint(dfs)
    # # # import ipdb; ipdb.set_trace()

    # # # 2
    # # # 4
    # # # 5
    # # # 8
    # # # 11
    # # # 12
