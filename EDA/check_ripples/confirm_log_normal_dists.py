#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-02-01 15:23:37 (ywatanabe)"

import mngs
import sys
sys.path.append(".")
import utils
import numpy as np

rips_df = utils.rips.load_rips(from_pkl=False) #extracts_firing_patterns=True) 
cons_df = utils.rips.load_cons_across_trials(from_pkl=False, only_correct=False)#extracts_firing_patterns=True)
assert len(rips_df) == len(cons_df) # 1170, 1581
cons_df = utils.rips.add_ripple_peak_amplitudeto_to_cons(cons_df)

# cons_df = mngs.io.load("./tmp/ripple_peak_amplitude_sd_added_cons_df.pkl")
cons_df["duration_ms"] = (cons_df.end_time - cons_df.start_time)*1000

dur_df = mngs.general.force_dataframe({
    "SWR": rips_df["duration_ms"],
    "Control": cons_df["duration_ms"].astype(float),    
})
mngs.io.save(dur_df, "./tmp/figs/hist/ripple_duration.csv")
mngs.io.save(np.log10(dur_df), "./tmp/figs/hist/log10_ripple_duration.csv")

amp_df = mngs.general.force_dataframe({
    "SWR": rips_df["ripple_peak_amplitude_sd"],
    "Control": cons_df["ripple_peak_amplitude_sd"],
})
mngs.io.save(amp_df, "./tmp/figs/hist/ripple_amplitude.csv")
mngs.io.save(np.log10(amp_df), "./tmp/figs/hist/log10_ripple_amplitude.csv")

