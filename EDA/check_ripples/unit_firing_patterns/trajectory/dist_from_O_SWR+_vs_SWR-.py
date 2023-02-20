#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-02-20 10:04:23 (ywatanabe)"
"""
Plots peri-SWR distance from O as a function of time from probe [s].
"""
import sys

sys.path.append(".")
import utils
import numpy as np
import mngs
from scipy.stats import zscore
import pandas as pd
    

def calc_dist_from_O(event_df):
    dists_rips = []
    for subject, roi in ROIs.items():
        subject = f"{subject:02d}"
        for session in ["01", "02"]:
            traj_session = mngs.io.load(
                f"./data/Sub_{subject}/Session_{session}/traj_z_by_session_{roi}.npy"
            )
            rips_mask = utils.rips.mk_rips_mask(
                event_df, subject, session, roi, 250
            )  # (49, 160)
            masked_traj_rips = traj_session * rips_mask[:, np.newaxis, :]
            norm_traj_rips = norm(masked_traj_rips, axis=1)
            dists_rips.append(norm_traj_rips)

    dists_rips = np.vstack(dists_rips)
    dists_rips[dists_rips == 0] = np.nan
            
    return dists_rips


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
    dists_rips = calc_dist_from_O(rips_df)
    dists_cons = calc_dist_from_O(cons_df)    
    # dists_rips[dists_rips == 0] = np.nan
    # dists_cons[dists_cons == 0] = np.nan    

    # dists_rips_smoothed = dists_rips
    # dists_cons_smoothed = dists_cons

    ns_rips = np.nansum(~np.isnan(dists_rips_smoothed), axis=0)
    ns_cons = np.nansum(~np.isnan(dists_cons_smoothed), axis=0)    
    # truncate = .3
    # sigma = 1
    # dists_rips_smoothed = np.vstack(
    #     [
    #         gaussian_filter1d(dists_rips[:, i_bin], sigma, truncate=truncate)
    #         for i_bin in range(dists_rips.shape[-1])
    #     ]
    # ).T
    # dists_cons_smoothed = np.vstack(
    #     [
    #         gaussian_filter1d(dists_cons[:, i_bin], sigma, truncate=truncate)            
    #         for i_bin in range(dists_cons.shape[-1])
    #     ]
    # ).T



    # Stats
    # # phases
    # dists = {}
    # ps = []
    # effs = []
    # for phase, (phase_bin_start, phase_bin_end) in PHASES_BINS_DICT.items():
    #     d1 = dists_cons[:, phase_bin_start:phase_bin_end]
    #     d2 = dists_rips[:, phase_bin_start:phase_bin_end]        

    #     d1 = d1[~np.isnan(d1)]
    #     d2 = d2[~np.isnan(d2)]

    #     dists[f"{phase}_cons"] = d1
    #     dists[f"{phase}_rips"] = d2        

    #     w, p, dof, eff = mngs.stats.brunner_munzel_test(d1, d2)
        
    #     ps.append(p)
    #     effs.append(eff)
    # print(np.array(ps).round(3)) # [0.004 0.    0.    0.   ]
    # print(np.array(effs).round(3)) # [0.004 0.    0.    0.   ]    
    # # dists = mngs.gen.force_dataframe(dists)
    # fig, ax = plt.subplots()
    # for i_col, col in enumerate(dists.keys()):
    #     ax.boxplot(
    #         dists[col],
    #         positions=[i_col],
    #         )
    # plt.show()

    # # gs
    # gs = {}
    # ps = []
    # effs = []
    # for phase, (phase_bin_start, phase_bin_end) in GS_BINS_DICT.items():
    #     d1 = dists_cons[:, phase_bin_start:phase_bin_end]        
    #     d2 = dists_rips[:, phase_bin_start:phase_bin_end]

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
    for i_bin in range(dists_rips.shape[-1]):
        d1 = dists_cons_smoothed[:, i_bin]
        d2 = dists_rips_smoothed[:, i_bin]        

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
    xx = np.linspace(0, 8, 160) - 6
    fig = plot(xx, dists_rips_smoothed, dists_cons_smoothed, is_sign, effs)#ns_rips)
    plt.show()    
    mngs.io.save(fig, "./tmp/figs/line/dist_from_O/SWR+_vs._SWR-.png")
    


    df = pd.DataFrame({
        "Time_s": xx,
        "SWR+ under": np.nanmean(dists_rips, axis=0) - np.nanstd(dists_rips, axis=0),
        "SWR+ mean": np.nanmean(dists_rips, axis=0),
        "SWR+ top": np.nanmean(dists_rips, axis=0) + np.nanstd(dists_rips, axis=0),
        "SWR- under": np.nanmean(dists_cons, axis=0) - np.nanstd(dists_cons, axis=0),
        "SWR- mean": np.nanmean(dists_cons, axis=0),
        "SWR- top": np.nanmean(dists_cons, axis=0) + np.nanstd(dists_cons, axis=0),
        "is_sign": is_sign*4,
    })
    mngs.io.save(df, "./tmp/figs/line/dist_from_O/SWR+_vs._SWR-.csv")    
