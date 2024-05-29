#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-03-22 15:15:06 (ywatanabe)"

import sys

sys.path.append(".")
import utils
import mngs
import numpy as np
import pandas as pd

# Functions
def get_gE_gR_session(events_df_s, subject, session, roi):
    def _get_g_session(events_df_s, subject, session, roi, phase_SWR):
        SWR_BINS = mngs.io.load("./config/global.yaml")["SWR_BINS"]        
        events_df_sp = events_df_s[events_df_s.phase == phase_SWR]

        traj = mngs.io.load(
            f"./data/Sub_{subject}/Session_{session}/traj_z_by_session_{roi}.npy"
        )

        traj_events = []
        for _, event in events_df_sp.iterrows():
            event_bin = int(event.center_time / (50 / 1000))
            _traj = traj[int((event.trial_number - 1))]
            _traj_event = _traj[
                :, event_bin + SWR_BINS["mid"][0] : event_bin + SWR_BINS["mid"][1]
            ]
            traj_events.append(_traj_event.T)
        traj_events = np.vstack(traj_events)

        # g = np.nanmedian(traj_events, axis=0)
        g = np.nanmean(traj_events, axis=0)        
        return g
    gE = _get_g_session(events_df_s, subject, session, roi, "Encoding")
    gR = _get_g_session(events_df_s, subject, session, roi, "Retrieval")
    return gE, gR

def add_geSWR_grSWR(events_df):
    events_df_all = []
    ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")

    for subject, roi in ROIs.items():
        subject = f"{subject:02d}"
        for session in ["01", "02"]:
            events_df_s = events_df[
                (events_df.subject == subject) * (events_df.session == session)
            ]

            gE, gR = get_gE_gR_session(events_df_s, subject, session, roi)

            events_df_s[f"geSWR"] = [
                gE for _ in range(len(events_df_s))
            ]
            events_df_s[f"grSWR"] = [
                gR for _ in range(len(events_df_s))
            ]
            events_df_all.append(events_df_s)
    events_df_all = pd.concat(events_df_all)
    return events_df_all

if __name__ == "__main__":
    rips_df = utils.rips.load_rips()
    events_df = add_geSWR_grSWR(rips_df)

