#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-02-24 10:23:31 (ywatanabe)"

import sys
sys.path.append(".")
import utils
import numpy as np
import mngs

def calc_inci(event_df):
    events_digi = []
    for subject, roi in ROIs.items():
        subject = f"{subject:02d}"
        for session in ["01", "02"]:
            event_df_session = event_df[(event_df.subject == subject) * (event_df.session == session)]
            events_digi_session = utils.rips.mk_events_mask(event_df_session, subject, session, roi, 0)
            if not len(event_df_session) == events_digi_session.sum():
                import ipdb; ipdb.set_trace()
            assert len(event_df_session) == events_digi_session.sum()
            events_digi.append(events_digi_session)
    events_digi = np.vstack(events_digi)            
    return events_digi

if __name__ == "__main__":
    # Loads
    rips_df = utils.rips.load_rips()
    cons_df = utils.rips.load_cons_across_trials()
    rips_df["center_time"] = (rips_df.start_time + rips_df.end_time).astype(float) / 2
    assert np.all(sorted(rips_df.center_time) == sorted(cons_df.center_time))

    ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")

    rips_digi = calc_inci(rips_df)
    cons_digi = calc_inci(cons_df)
    print(rips_digi.sum())
    print(cons_digi.sum())

    rips_digi.sum(axis=0) == cons_digi.sum(axis=0)

    subject = "03"
    session = "01"
    n_trials = len(mngs.io.load(f"./data/Sub_{subject}/Session_{session}/trials_info.csv"))    
    rips_df[(rips_df.subject == "03") * (rips_df.session == "01")]
