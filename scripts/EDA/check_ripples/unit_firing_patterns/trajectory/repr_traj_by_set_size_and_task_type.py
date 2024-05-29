#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-05-25 20:06:03 (ywatanabe)"

import matplotlib

matplotlib.use("TkAgg")

import mngs
import numpy as np
import pandas as pd


def collect_trajectories(set_size, match):
    indi = (meta.set_size == set_size) * (meta.match == match)

    trajs = trajs_all[indi]
    trajs_session_x = {}
    trajs_session_y = {}
    trajs_session_z = {}
    gs = {}
    for phase, (ss, ee) in GS_BINS_DICT.items():
        trajs_session_x[phase] = np.median(trajs, axis=0)[:, ss:ee][0]
        trajs_session_y[phase] = np.median(trajs, axis=0)[:, ss:ee][1]
        trajs_session_z[phase] = np.median(trajs, axis=0)[:, ss:ee][2]
        gs[phase] = np.nanmedian(
            trajs.transpose(1, 0, 2)[:, :, ss:ee].reshape(3, -1),
            axis=-1,
        )

    # session trajectory
    df_x = mngs.gen.force_dataframe(trajs_session_x)
    df_y = mngs.gen.force_dataframe(trajs_session_y)
    df_z = mngs.gen.force_dataframe(trajs_session_z)
    df = pd.concat([df_x, df_y, df_z], axis=1)

    # gs
    df_gs = pd.DataFrame(gs).T

    return df, df_gs


def sort_df(df):
    _df = df.copy()
    df_sorted = []
    for phase in PHASES:
        df_sorted.append(_df.iloc[:, _df.columns == phase])
    return pd.concat(df_sorted, axis=1)


def sort_df_gs(df_gs):
    df_gs.columns = ["x", "y", "z"]
    df_gs_sorted = []
    for phase in PHASES:
        df_gs_phase = pd.DataFrame(df_gs.T[phase]).T
        df_gs_phase.columns = [f"{phase}_{col}" for col in df_gs.columns]
        df_gs_phase = df_gs_phase.reset_index()
        del df_gs_phase["index"]
        df_gs_sorted.append(df_gs_phase)
    return pd.concat(df_gs_sorted, axis=1)


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

    ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")
    # bin_size = 200 * pq.ms
    # n_bins = int((8 / bin_size.rescale("s")).magnitude)

    subject = "06"
    session = "02"
    roi = ROIs[int(subject)]

    LPATH = (
        f"./data/Sub_{subject}/Session_{session}/traj_z_by_session_{roi}.npy"
    )
    trajs = mngs.io.load(LPATH)
    LPATH = f"./data/Sub_{subject}/Session_{session}/trials_info.csv"
    meta = mngs.io.load(LPATH)

    trajs_all = trajs.copy()

    for set_size in [4, 6, 8]:
        for match in [1, 2]:

            df, df_gs = collect_trajectories(set_size, match)

            df = sort_df(df)
            df_gs = sort_df_gs(df_gs)
            df_merged = pd.concat([df_gs, df], axis=1)

            spath = (
                f"./res/figs/scatter/repr_traj/session_traj_Subject_{subject}_Session_{session}"
                f"_set_size_{set_size}_match_{match}.csv"
            )
            mngs.io.save(df_merged, spath)
