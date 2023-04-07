#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-03-15 12:24:17 (ywatanabe)"
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


def calc_dist_from_P(event_df, base_phase=None):
    dists_rips = []
    for subject, roi in ROIs.items():
        subject = f"{subject:02d}"
        for session in ["01", "02"]:
            traj_session = mngs.io.load(
                f"./data/Sub_{subject}/Session_{session}/traj_z_by_session_{roi}.npy"
            )
            if base_phase is not None:
                coords_gP = np.nanmedian(
                    traj_session[:, :, GS_BINS_DICT[base_phase][0] : GS_BINS_DICT[base_phase][1]],
                    axis=-1,
                    keepdims=True,
                )
                traj_session -= coords_gP
            rips_mask = utils.rips.mk_events_mask(
                event_df, subject, session, roi, 250
            )  # (49, 160)
            masked_traj_rips = traj_session * rips_mask[:, np.newaxis, :]
            norm_traj_rips = norm(masked_traj_rips, axis=1)
            dists_rips.append(norm_traj_rips)

    dists_rips = np.vstack(dists_rips)
    dists_rips[dists_rips == 0] = np.nan

    return dists_rips


def plot(xx, dists_rips, dists_cons, is_sign, effs, base_phase):
    fig, ax = plt.subplots()
    # ax.plot(xx, np.array(effs) * 10, label="effs")
    ax = mngs.plt.ax_fill_between(
        ax, xx, np.nanmean(dists_rips, axis=0), np.nanstd(dists_rips, axis=0), "SWR+"
    )
    ax = mngs.plt.ax_fill_between(
        ax, xx, np.nanmean(dists_cons, axis=0), np.nanstd(dists_cons, axis=0), "SWR-"
    )
    ax.scatter(xx[is_sign], 4 * np.ones(is_sign.sum()))
    ax.legend()
    [ax.axvline(x=_x, linestyle="--", color="gray") for _x in [-5, -3, 0, 2]]
    ax.set_ylabel(f"Peri-SWR distance from {base_phase} [a.u.]")
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

    base_phase = None
    rips_df_m = rips_df
    cons_df_m = cons_df    
    ###
    # match = 1
    # set_size = 8
    # rips_df_m = rips_df[(rips_df.match == match)*(rips_df.set_size == set_size)]
    # cons_df_m = cons_df[(cons_df.match == match)*(cons_df.set_size == set_size)]
    # )
    # *(rips_df.set_size == set_size)###
    ##*(cons_df.set_size == set_size#
    
    (
        PHASES,
        PHASES_BINS_DICT,
        GS_BINS_DICT,
        COLORS_DICT,
        BIN_SIZE,
    ) = utils.define_phase_time()

    # Distance from P (O, F, E, M, or R) during SWR+/SWR-
    dists_rips = calc_dist_from_P(rips_df_m, base_phase)
    dists_cons = calc_dist_from_P(cons_df_m, base_phase)

    dists_rips_smoothed = dists_rips
    dists_cons_smoothed = dists_cons
    
    ns_rips = np.nansum(~np.isnan(dists_rips_smoothed), axis=0)
    ns_cons = np.nansum(~np.isnan(dists_cons_smoothed), axis=0)

    mm_rips, sd_rips = np.nanmean(dists_rips, axis=0), np.nanstd(dists_rips, axis=0)
    mm_cons, sd_cons = np.nanmean(dists_cons, axis=0), np.nanstd(dists_cons, axis=0)

    ci_rips = 1.96 * sd_rips/ns_rips
    ci_cons = 1.96 * sd_cons/ns_cons    

    # BM test per bin
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
    fig = plot(xx, dists_rips_smoothed, dists_cons_smoothed, is_sign, effs, base_phase)  # ns_rips)
    plt.show()
    mngs.io.save(fig, f"./tmp/figs/line/dist_from_P_tc/SWR+_vs._SWR-_from_{base_phase}.png")

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
    mngs.io.save(df, f"./tmp/figs/line/dist_from_P_tc/SWR+_vs._SWR-_from_{base_phase}.csv")
