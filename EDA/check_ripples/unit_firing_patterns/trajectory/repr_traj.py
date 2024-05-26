#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-05-25 21:08:01 (ywatanabe)"

import matplotlib

matplotlib.use("TkAgg")

import mngs
import numpy as np
import pandas as pd

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

    __import__("ipdb").set_trace()

    meta.set_size
    meta.match

    trajs_session_x = {}
    trajs_session_y = {}
    trajs_session_z = {}
    gs = {}
    for phase, (ss, ee) in GS_BINS_DICT.items():
        trajs_session_x[phase] = np.median(trajs, axis=0)[:, ss:ee][0]
        trajs_session_y[phase] = np.median(trajs, axis=0)[:, ss:ee][1]
        trajs_session_z[phase] = np.median(trajs, axis=0)[:, ss:ee][2]
        gs[phase] = np.nanmedian(
            trajs.transpose(1, 0, 2)[:, :, ss:ee].reshape(3, -1), axis=-1
        )

    # session trajectory
    df_x = mngs.gen.force_dataframe(trajs_session_x)
    df_y = mngs.gen.force_dataframe(trajs_session_y)
    df_z = mngs.gen.force_dataframe(trajs_session_z)
    df = pd.concat([df_x, df_y, df_z], axis=1)
    mngs.io.save(
        df,
        "./res/figs/scatter/repr_traj/session_traj_Subject_06_Session_02.csv",
    )

    # gs
    df_gs = pd.DataFrame(gs).T
    mngs.io.save(
        df_gs,
        "./res/figs/scatter/repr_traj/session_gs_Subject_06_Session_02.csv",
    )
