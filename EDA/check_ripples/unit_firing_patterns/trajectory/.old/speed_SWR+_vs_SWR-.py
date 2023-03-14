#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-02-24 12:44:13 (ywatanabe)"
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


def calc_speed_from_O(event_df):
    speeds_rips = []
    for subject, roi in ROIs.items():
        subject = f"{subject:02d}"
        for session in ["01", "02"]:
            traj_session = mngs.io.load(
                f"./data/Sub_{subject}/Session_{session}/traj_z_by_session_{roi}.npy"
            )
            rips_mask = utils.rips.mk_events_mask(
                event_df, subject, session, roi, 250
            )  # (49, 160)
            masked_traj_rips = traj_session * rips_mask[:, np.newaxis, :]
            diff_traj_rips = masked_traj_rips[..., :-1] - masked_traj_rips[..., 1:]
            speed_traj_rips = norm(diff_traj_rips, axis=1)
            # import ipdb; ipdb.set_trace()
            # norm_traj_rips = norm(masked_traj_rips, axis=1)
            speeds_rips.append(speed_traj_rips)

    speeds_rips = np.vstack(speeds_rips)
    speeds_rips[speeds_rips == 0] = np.nan

    return speeds_rips


def plot(xx, dists_rips, dists_cons, is_sign, effs):
    fig, ax = plt.subplots()
    ax.plot(xx, np.array(effs) * 10, label="effs")
    ax = mngs.plt.ax_fill_between(
        ax, xx, np.nanmean(dists_rips, axis=0), np.nanstd(dists_rips, axis=0), "SWR+"
    )
    ax = mngs.plt.ax_fill_between(
        ax, xx, np.nanmean(dists_cons, axis=0), np.nanstd(dists_cons, axis=0), "SWR-"
    )
    ax.scatter(xx[is_sign], 4 * np.ones(is_sign.sum()))
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

    # Speedance from O during SWR+/SWR-
    speeds_rips = calc_speed_from_O(rips_df)
    speeds_cons = calc_speed_from_O(cons_df)

    speeds_rips_smoothed = speeds_rips
    speeds_cons_smoothed = speeds_cons
    
    ns_rips = np.nansum(~np.isnan(speeds_rips_smoothed), axis=0)
    ns_cons = np.nansum(~np.isnan(speeds_cons_smoothed), axis=0)

    mm_rips, sd_rips = np.nanmean(speeds_rips, axis=0), np.nanstd(speeds_rips, axis=0)
    mm_cons, sd_cons = np.nanmean(speeds_cons, axis=0), np.nanstd(speeds_cons, axis=0)

    ci_rips = 1.96 * sd_rips/ns_rips
    ci_cons = 1.96 * sd_cons/ns_cons    

    # BM test per bin
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
    fig = plot(xx, speeds_rips_smoothed, speeds_cons_smoothed, is_sign, effs)  # ns_rips)
    plt.show()
    mngs.io.save(fig, "./tmp/figs/line/speed/SWR+_vs._SWR-.png")

    # to csv
    df = pd.DataFrame(
        {
            "Time_s": xx,
            "SWR+ under": mm_rips - ci_rips,
            "SWR+ mean": mm_rips,
            "SWR+ upper": mm_rips + ci_rips,
            "SWR- under": mm_cons - ci_cons,
            "SWR- mean": mm_cons,
            "SWR- upper": mm_cons + ci_cons,
            "is_sign": is_sign * 4,
        }
    )
    mngs.io.save(df, "./tmp/figs/line/speed/SWR+_vs._SWR-.csv")
