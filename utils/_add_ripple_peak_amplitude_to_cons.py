#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-02-01 15:03:11 (ywatanabe)"

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-01-28 15:33:41 (ywatanabe)"

import sys

sys.path.append(".")
from siEEG_ripple import utils
import torch
import warnings
import numpy as np
from tqdm import tqdm
import mngs



def add_ripple_peak_amplitudeto_to_cons(cons_df):

    LOW_HZ = 80
    HIGH_HZ = 140
    iEEG_SAMP_RATE = 2000


    ripple_peak_amplitude_sd_all = []
    for i_con, con_df in tqdm(cons_df.iterrows()):
        i_trial = int(con_df.trial_number - 1)
        subject = f"{int(con_df.subject):02d}"
        session = f"{int(con_df.session):02d}"
        roi = con_df.ROI
        start_pts = int(con_df.start_time * iEEG_SAMP_RATE)
        end_pts = int(con_df.end_time * iEEG_SAMP_RATE)

        iEEG, iEEG_common_ave = utils.load_iEEG(
            subject, session, roi, return_common_averaged_signal=True
        )

        iEEG = iEEG[i_trial]

        # bandpass filtering
        iEEG_ripple_band_passed = np.array(
            mngs.dsp.bandpass(
                torch.tensor(np.array(iEEG).astype(np.float32)),
                iEEG_SAMP_RATE,
                low_hz=LOW_HZ,
                high_hz=HIGH_HZ,
            )
        )
        ripple_band_iEEG_traces = iEEG_ripple_band_passed[:, start_pts:end_pts]
        ripple_peak_amplitude = np.abs(ripple_band_iEEG_traces).max(axis=1).squeeze()
        ripple_band_baseline_sd = ripple_band_iEEG_traces.std(axis=1).squeeze()
        ripple_peak_amplitude_sd = (ripple_peak_amplitude / ripple_band_baseline_sd).mean()

        ripple_peak_amplitude_sd_all.append(ripple_peak_amplitude_sd)

    cons_df["ripple_peak_amplitude_sd"] = ripple_peak_amplitude_sd_all

    return cons_df


if __name__ == "__main__":
    cons_df = utils.rips.load_cons_across_trials()
    cons_df = add_ripple_peak_amplitude(cons_df)
    mngs.io.save(cons_df, "./tmp/ripple_peak_amplitude_sd_added_cons_df.pkl")
