#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-02-20 09:47:10 (ywatanabe)"

import sys

sys.path.append(".")
import utils
import numpy as np
import mngs
from scipy.stats import zscore

def calc_speed(rips_df):
    # dists_rips = []
    speeds_rips = []
    for subject, roi in ROIs.items():
        subject = f"{subject:02d}"
        for session in ["01", "02"]:
            """
            subject = "01"
            session = "01"
            roi = "AHL"
            """            
            traj_session = mngs.io.load(
                f"./data/Sub_{subject}/Session_{session}/traj_z_by_session_{roi}.npy"
            )
            speed_session = norm(traj_session[..., 1:] - traj_session[..., :-1], axis=1)
            rips_digi = utils.rips.to_digi_rips(
                rips_df, subject, session, roi
            )  # (49, 160)
            masked_speed_rips = speed_session * rips_digi[..., :-1]            
            # masked_traj_rips = traj_session * rips_digi[:, np.newaxis, :]
            # norm_traj_rips = norm(masked_traj_rips, axis=1)
            
            # mm, ss = norm_traj_all_non_zero.mean(), norm_traj_all_non_zero.std()

            # norm_traj_rips = (norm_traj_rips - mm) / ss
            # norm_traj_cons = (norm_traj_cons - mm) / ss            
            
            # norm_traj_rips_z = norm_traj_rips
            # norm_traj_rips_z[norm_traj_rips_z != 0] = zscore(norm_traj_rips_z[norm_traj_rips_z != 0])
            # dists_rips.append(norm_traj_rips)
            speeds_rips.append(masked_speed_rips)

    speeds_rips = np.vstack(speeds_rips)
    speeds_rips[speeds_rips == 0] = np.nan
            
    return speeds_rips


def plot(xx, dists_rips, dists_cons, is_sign, effs):
    fig, ax = plt.subplots()
    ax.plot(xx, np.array(effs)*10, label="effs")
    ax = mngs.plt.ax_fill_between(
        ax, xx, np.nanmean(dists_rips, axis=0), np.nanstd(dists_rips, axis=0), "SWR+"
    )
    ax = mngs.plt.ax_fill_between(
        ax, xx, np.nanmean(dists_cons, axis=0), np.nanstd(dists_cons, axis=0), "SWR-"        
    )
    ax.scatter(xx[is_sign], 4*np.ones(is_sign.sum()))
    ax.legend()
    [ax.axvline(x=_x, linestyle="--", color="gray") for _x in [-5, -3, 0, 2]]
    ax.set_ylabel("Peri-SWR distance from O [a.u.]")
    ax.set_xlabel("Time from probe [s]")    
    return fig


if __name__ == "__main__":
    from scipy.linalg import norm
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d
    
    # Loads
    ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")
    rips_df = utils.rips.load_rips()
    cons_df = utils.rips.load_cons_across_trials()
    (
        PHASES,
        PHASES_BINS_DICT,
        GS_BINS_DICT,
        COLORS_DICT,
        BIN_SIZE,
    ) = utils.define_phase_time()

    # Trajectories during SWR+/SWR-
    speeds_rips = calc_speed(rips_df)
    speeds_cons = calc_speed(cons_df)        

    speeds_rips_smoothed = speeds_rips
    speeds_cons_smoothed = speeds_cons

    ns_rips = np.nansum(~np.isnan(speeds_rips_smoothed), axis=0)
    ns_cons = np.nansum(~np.isnan(speeds_cons_smoothed), axis=0)    
    # truncate = .3
    # sigma = 1
    # speeds_rips_smoothed = np.vstack(
    #     [
    #         gaussian_filter1d(speeds_rips[:, i_bin], sigma, truncate=truncate)
    #         for i_bin in range(speeds_rips.shape[-1])
    #     ]
    # ).T
    # speeds_cons_smoothed = np.vstack(
    #     [
    #         gaussian_filter1d(speeds_cons[:, i_bin], sigma, truncate=truncate)            
    #         for i_bin in range(speeds_cons.shape[-1])
    #     ]
    # ).T



    # Stats
    # # phases
    # speeds = {}
    # ps = []
    # effs = []
    # for phase, (phase_bin_start, phase_bin_end) in PHASES_BINS_DICT.items():
    #     d1 = speeds_cons[:, phase_bin_start:phase_bin_end]
    #     d2 = speeds_rips[:, phase_bin_start:phase_bin_end]        

    #     d1 = d1[~np.isnan(d1)]
    #     d2 = d2[~np.isnan(d2)]

    #     speeds[f"{phase}_cons"] = d1
    #     speeds[f"{phase}_rips"] = d2        

    #     w, p, dof, eff = mngs.stats.brunner_munzel_test(d1, d2)
        
    #     ps.append(p)
    #     effs.append(eff)
    # print(np.array(ps).round(3)) # [0.004 0.    0.    0.   ]
    # print(np.array(effs).round(3)) # [0.004 0.    0.    0.   ]    
    # # speeds = mngs.gen.force_dataframe(speeds)
    # fig, ax = plt.subplots()
    # for i_col, col in enumerate(speeds.keys()):
    #     ax.boxplot(
    #         speeds[col],
    #         positions=[i_col],
    #         )
    # plt.show()

    # # gs
    # gs = {}
    # ps = []
    # effs = []
    # for phase, (phase_bin_start, phase_bin_end) in GS_BINS_DICT.items():
    #     d1 = speeds_cons[:, phase_bin_start:phase_bin_end]        
    #     d2 = speeds_rips[:, phase_bin_start:phase_bin_end]

    #     d1 = d1[~np.isnan(d1)]
    #     d2 = d2[~np.isnan(d2)]

    #     gs[f"{phase}_cons"] = d1        
    #     gs[f"{phase}_rips"] = d2

    #     w, p, dof, eff = mngs.stats.brunner_munzel_test(d1, d2)
        
    #     ps.append(p)
    #     effs.append(eff)
    # ps = np.array(ps)
    # effs = np.array(effs)    
    # print(ps.round(3)) # [0.03  0.    0.001 0.003]
    # print(effs.round(3)) # [0.435 0.294 0.388 0.367]
    # # df = mngs.gen.force_dataframe(gs)
    # # import seaborn as sns
    # fig, ax = plt.subplots()
    # for i_xx, key in enumerate(gs.keys()):
    #     ax.boxplot(gs[key], positions=[i_xx])
    # plt.show()
    
    
    # bins
    ps = []
    effs = []
    for i_bin in range(speeds_rips.shape[-1]):
        d1 = speeds_cons_smoothed[:, i_bin]
        d2 = speeds_rips_smoothed[:, i_bin]        

        d1 = d1[~np.isnan(d1)]
        d2 = d2[~np.isnan(d2)]        

        try:
            w, p, dof, eff = mngs.stats.brunner_munzel_test(d1, d2)
        except Exception as e:
            print(e)
            p = np.nan
            eff = np.nan
        ps.append(p)
        effs.append(eff)

    print(np.array(ps).round(3))
    is_sign = np.array(ps) < 0.05

    # Plots
    xx = (np.linspace(0, 8, 160) - 6)[:-1]
    fig = plot(xx, speeds_rips_smoothed, speeds_cons_smoothed, is_sign, effs)#ns_rips)
    plt.show()    
    mngs.io.save(fig, "./tmp/figs/line/speed/SWR+_vs._SWR-.png")
    

    import pandas as pd
    df = pd.DataFrame({
        "Time_s": xx,
        "SWR+ under": np.nanmean(speeds_rips, axis=0) - np.nanstd(speeds_rips, axis=0),
        "SWR+ mean": np.nanmean(speeds_rips, axis=0),
        "SWR+ top": np.nanmean(speeds_rips, axis=0) + np.nanstd(speeds_rips, axis=0),
        "SWR- under": np.nanmean(speeds_cons, axis=0) - np.nanstd(speeds_cons, axis=0),
        "SWR- mean": np.nanmean(speeds_cons, axis=0),
        "SWR- top": np.nanmean(speeds_cons, axis=0) + np.nanstd(speeds_cons, axis=0),
        "is_sign": is_sign*4,
    })
    mngs.io.save(df, "./tmp/figs/line/speed/SWR+_vs._SWR-.csv")    
