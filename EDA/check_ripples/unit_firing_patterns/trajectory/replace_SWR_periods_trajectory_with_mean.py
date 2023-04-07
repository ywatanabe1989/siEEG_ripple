#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-03-16 17:27:37 (ywatanabe)"

import mngs
import sys

sys.path.append(".")
import utils
import numpy as np

rips_df = utils.rips.load_rips()

z_by = "by_session"
ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")
for subject, roi in ROIs.items():
    for session in ["01", "02"]:
        subject = f"{int(subject):02d}"
        lpath = f"./data/Sub_{subject}/Session_{session}/traj_z_{z_by}_{roi}.npy"
        trajs = mngs.io.load(lpath)
        trajs_medians = np.median(trajs, axis=-1, keepdims=True)
        rips_df_s = rips_df[(rips_df.subject == subject) * (rips_df.session == session)]
        for _, rip in rips_df_s.iterrows():
            center_bin = int(rip.center_time / (50 / 1000))
            trajs[int(rip.trial_number-1),:,center_bin-5:center_bin+6] = \
                trajs_medians[int(rip.trial_number-1)]
        mngs.io.save(trajs, lpath.replace(".npy", "_replaced_SWR_periods_with_median.npy"))
