#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-02-07 08:26:17 (ywatanabe)"


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

sys.path.append(".")
import utils

# Functions
# def plot_traj(rips_df, is_control=False, set_size=None, match=None):

#     events_str = "cons" if is_control else "rips"

#     if set_size is not None:
#         rips_df = rips_df[rips_df.set_size == set_size]
#     set_size_str = f"_set_size_{set_size}" if set_size is not None else ""

#     if match is not None:
#         rips_df = rips_df[rips_df.match == match]
#     match_str = f"_match_{match}" if match is not None else ""

#     fig, axes = plt.subplots(
#         ncols=n_bases, sharex=True, sharey=True, figsize=(6.4 * 3, 4.8 * 3)
#     )  # sharey=True,
#     xlim = (-500, 500)
#     out_df = pd.DataFrame()
#     for i_base, (cbs, cbe) in enumerate(cols_base_starts_ends):
#         dir_txt = f"{cbs}-{cbe}_based"

#         ax = axes[i_base]
#         i_ax = i_base

#         samp_m = mngs.general.listed_dict(PHASES)
#         samp_s = mngs.general.listed_dict(PHASES)

#         for i_phase, phase in enumerate(PHASES):

#             rips_df_phase = rips_df[rips_df.phase == phase]

#             centers_ms = []
#             for delta_bin in range(-39, 39):
#                 col1 = f"{delta_bin-1}"
#                 col1_str = f"{int((delta_bin-1)*BIN_SIZE.magnitude)}"
#                 col2 = f"{delta_bin}"
#                 col2_str = f"{int((delta_bin)*BIN_SIZE.magnitude)}"

#                 centers_ms.append(int((delta_bin - 0.5) * BIN_SIZE.magnitude))

#                 # gets vectors
#                 v = np.vstack(rips_df_phase[col2]) - np.vstack(rips_df_phase[col1])
#                 if cbs in PHASES:  # basis transformation
#                     v_base = np.vstack(rips_df_phase[cbe]) - np.vstack(
#                         rips_df_phase[cbs]
#                     )
#                     v_rebased = [
#                         mngs.linalg.rebase_a_vec(v[ii], v_base[ii])
#                         for ii in range(len(v))
#                     ]
#                     # v_rebased = np.log10(np.abs(v_rebased)) # fixme
#                     v_rebased = np.abs(v_rebased)  # fixme

#                 if cbs is None:  # just the norm
#                     v_rebased = [mngs.linalg.nannorm(v[ii]) for ii in range(len(v))]
#                     v_rebased = np.abs(v_rebased)  # fixme

#                 elif "vec_4_8" in cbs:
#                     v_base = rips_df_phase[cbs]
#                     v_rebased = [
#                         mngs.linalg.rebase_a_vec(v[ii], v_base[ii])
#                         for ii in range(len(v))
#                     ]
#                     v_rebased = np.abs(v_rebased)  # fixme

#                 mm, ss = mngs.gen.describe(v_rebased, method="mean")

#                 samp_m[phase].append(mm)
#                 samp_s[phase].append(ss)

#                 # nan_indi = np.isnan(v_rebased)
#                 # n_samp = (~nan_indi).sum()
#                 # v_rebased = v_revased[nan_indi]

#                 # samp_m[phase].append(np.nanmean(v_rebased))
#                 # samp_s[phase].append(np.nanstd(v_rebased) / 3)

#             ax.axhline(y=0, xmin=xlim[0], xmax=xlim[1], linestyle="--", color="gray")

#             ax.errorbar(
#                 x=np.array(centers_ms) + i_phase * 3,
#                 y=samp_m[phase],
#                 yerr=samp_s[phase],
#                 label=phase,
#             )
#             ax.legend(loc="upper right")

#         ax.set_xlim(xlim)

#         # ylim = (-1.25, 0.25) # log10
#         ylim = (0, 3)
#         if i_ax != 0:
#             title = dir_txt.replace("_based", "").replace("-", " -> ")
#             ax.set_ylim(ylim)
#             # ax.set_ylim(-0.3, 1)
#         else:
#             title = "Speed (norm)"
#             ax.set_ylim(ylim)
#             # ax.set_ylim(-0.3, 2)

#         ax.set_title(title)

#         samp_m = pd.DataFrame(samp_m)
#         samp_m.columns = [f"{col}_{cbs}-{cbe}_med" for col in samp_m.columns]
#         samp_s = pd.DataFrame(samp_s)
#         samp_s.columns = [f"{col}_{cbs}-{cbe}_s" for col in samp_s.columns]

#         # out_dict[f"{cbs}-{cbe}_med"] = samp_m
#         # out_dict[f"{cbs}-{cbe}_se"] = samp_s

#         out_df = pd.concat([out_df, pd.concat([samp_m, samp_s], axis=1)], axis=1)

#     fig.suptitle(f"Set size: {set_size}\nMatch: {match}")
#     fig.supylabel("Speed")
#     fig.supxlabel("Time from SWR [ms]")
#     # plt.show()
#     mngs.io.save(
#         fig,
#         f"./tmp/figs/hist/traj_speed/{match_str}/all_{events_str}{set_size_str}.png",
#     )
#     mngs.io.save(
#         out_df,
#         f"./tmp/figs/hist/traj_speed/{match_str}/all_{events_str}{set_size_str}.csv",
#     )
#     return fig


if __name__ == "__main__":
    import mngs
    import numpy as np

    # matplotlib.use("Agg")

    # Parameters
    (
        PHASES,
        PHASES_BINS_DICT,
        GS_BINS_DICT,
        COLORS_DICT,
        BIN_SIZE,
    ) = utils.define_phase_time()
    ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")

    # Loads data
    rips_df = utils.rips.add_coordinates(
        utils.rips.load_rips_df_with_traj(BIN_SIZE, is_control=False)
    )
    cons_df = utils.rips.add_coordinates(
        utils.rips.load_rips_df_with_traj(BIN_SIZE, is_control=True)
    )

    subject = "06"
    session = "02"

    rips_df = rips_df[(rips_df.subject == subject) * (rips_df.session == session)]
    cons_df = cons_df[(cons_df.subject == subject) * (cons_df.session == session)]
    cons_df = cons_df[: len(rips_df)]
    assert len(rips_df) == len(cons_df)

    rips_trajs, cons_trajs = [], []
    for ii in range(-11, 12):
        rips_coords_ii = np.vstack(rips_df[f"{ii}"]).astype(float)
        rips_coords_ii = rips_coords_ii[~np.isnan(rips_coords_ii).any(axis=-1)]
        rips_trajs.append(rips_coords_ii)

        cons_coords_ii = np.vstack(cons_df[f"{ii}"]).astype(float)
        cons_coords_ii = cons_coords_ii[~np.isnan(cons_coords_ii).any(axis=-1)]
        cons_trajs.append(cons_coords_ii)

    rips_trajs_means = np.array([rt.mean(axis=0) for rt in rips_trajs])
    rips_trajs_stds = np.array([rt.std(axis=0) for rt in rips_trajs])

    cons_trajs_means = np.array([rt.mean(axis=0) for rt in cons_trajs])
    cons_trajs_stds = np.array([rt.std(axis=0) for rt in cons_trajs])

    xx = np.arange(-11, 12) * 50
    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
    [
        axes[0].errorbar(xx, cons_trajs_means[:, ii], yerr=cons_trajs_stds[:, ii], label=ii)
        for ii in range(cons_trajs_means.shape[-1])
    ]
    [
        axes[1].errorbar(xx, rips_trajs_means[:, ii], yerr=rips_trajs_stds[:, ii], label=ii)
        for ii in range(rips_trajs_means.shape[-1])
    ]
    axes[0].legend()
    axes[1].legend()    
    
    plt.show()

    df = pd.DataFrame({
        "x": xx,
        "SWR_factor_1_under": rips_trajs_means[:, 0] - rips_trajs_stds[:,0],
        "SWR_factor_1_mean": rips_trajs_means[:, 0],
        "SWR_factor_1_top": rips_trajs_means[:, 0] + rips_trajs_stds[:,0],        
        "SWR_factor_2_under": rips_trajs_means[:, 1] - rips_trajs_stds[:,1],
        "SWR_factor_2_mean": rips_trajs_means[:, 1],
        "SWR_factor_2_top": rips_trajs_means[:, 1] + rips_trajs_stds[:,1],        
        "SWR_factor_3_under": rips_trajs_means[:, 2] - rips_trajs_stds[:,2],
        "SWR_factor_3_mean": rips_trajs_means[:, 2],
        "SWR_factor_3_top": rips_trajs_means[:, 2] + rips_trajs_stds[:,2],
        
        "Control_factor_1_under": cons_trajs_means[:, 0] - cons_trajs_stds[:,0],
        "Control_factor_1_mean": cons_trajs_means[:, 0],
        "Control_factor_1_top": cons_trajs_means[:, 0] + cons_trajs_stds[:,0],        
        "Control_factor_2_under": cons_trajs_means[:, 1] - cons_trajs_stds[:,1],
        "Control_factor_2_mean": cons_trajs_means[:, 1],
        "Control_factor_2_top": cons_trajs_means[:, 1] + cons_trajs_stds[:,1],        
        "Control_factor_3_under": cons_trajs_means[:, 2] - cons_trajs_stds[:,2],
        "Control_factor_3_mean": cons_trajs_means[:, 2],
        "Control_factor_3_top": cons_trajs_means[:, 2] + cons_trajs_stds[:,2],        
    })
    mngs.io.save(df, "./tmp/figs/line/peri_ripple_traj.csv")
