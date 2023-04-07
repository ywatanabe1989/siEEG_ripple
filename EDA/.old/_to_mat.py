#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-02-15 15:47:59 (ywatanabe)"

import mngs
from glob import glob
import sys
sys.path.append(".")
import utils
import numpy as np

def save_mat_traj(subject, session, roi):
    lpath_traj = f"./data/Sub_{subject}/Session_{session}/traj_z_by_session_{roi}.npy"
    spath_traj = lpath_traj.replace("./data/", "./data/mat/").replace(".npy", ".mat")
    traj = mngs.io.load(lpath_traj)
    mngs.io.save({"trajectory": traj}, spath_traj)
    # mngs.io.load("./test.mat")

def save_mat_ripples(rips_df, subject, session, roi):
    spath_ripples = f"./data/mat/Sub_{subject}/Session_{session}/ripples_{roi}.mat"    
    rips_df_session = rips_df[(rips_df.subject == subject) * (rips_df.session == session)]
    rips_df_session = rips_df_session\
        [["subject", "session", "trial_number", "start_time", "center_time", "end_time", "set_size", "match"]]
    mngs.io.save({"ripples": rips_df_session}, spath_ripples)

    # ripples digi
    n_trials = len(mngs.io.load(f"./data/Sub_{subject}/Session_{session}/trials_info.csv"))
    bin_s = 50 / 1000
    n_bins = int(8 / bin_s)
    rips_digi = np.zeros([n_trials, n_bins], dtype=int)
    for i_trial in range(n_trials):
        print(i_trial)
        rips_df_trial = rips_df_session[rips_df_session.trial_number == i_trial+1]
        for i_rip, (_, rip) in enumerate(rips_df_trial.iterrows()):
            start_bin = int(rip.start_time / bin_s)
            end_bin = int(rip.end_time / bin_s)
            rips_digi[i_trial, start_bin:end_bin] = 1
    mngs.io.save({"ripples": rips_digi}, spath_ripples.replace("ripples", "ripples_digi"))    


def main():
    rips_df = utils.rips.load_rips()

    ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")
    for subject, roi in ROIs.items():
        subject = f"{subject:02d}"    
        for session in range(1,3):
            session = f"{session:02d}"
            save_mat_traj(subject, session, roi)
            save_mat_ripples(rips_df, subject, session, roi)

_sts = mngs.io.load("./data/Sub_01/Session_01/spike_times_AHL.pkl")
sts = []
for st in _sts:
    print()
    st.columns = [col.replace("Spike_Times_", "")for col in st.columns]
    sts.append(st)
import pandas as pd

mngs.io.save({"spike_times": sts}, './data/mat/spike_times_Sub_01_Session_01.mat')
if __name__ == "__main__":
    main()
