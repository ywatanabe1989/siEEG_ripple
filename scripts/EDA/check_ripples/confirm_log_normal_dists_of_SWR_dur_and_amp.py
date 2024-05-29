#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-02-27 07:49:07 (ywatanabe)"

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
mngs.gen.describe(np.array(dur_df["SWR"]), method="median")
mngs.gen.describe(np.array(dur_df["Control"]), method="median")
# mngs.io.save(dur_df, "./tmp/figs/hist/ripple_duration.csv")
# dur_df = mngs.io.load("./tmp/figs/hist/ripple_duration.csv")
# mngs.gen.describe(np.array(dur_df["SWR"]), "median")
# mngs.gen.describe(np.array(dur_df["Control"]), "median")
mngs.io.save(np.log10(dur_df), "./tmp/figs/hist/log10_ripple_duration.csv")




amp_df = mngs.general.force_dataframe({
    "SWR": rips_df["ripple_peak_amplitude_sd"],
    "Control": cons_df["ripple_peak_amplitude_sd"],
})
w, pval_bm, dof, effsize = mngs.stats.brunner_munzel_test(amp_df["SWR"], amp_df["Control"])
mngs.gen.describe(np.array(amp_df["SWR"]), method="median")
mngs.gen.describe(np.array(amp_df["Control"]), method="median")
mngs.io.save(amp_df, "./tmp/figs/hist/ripple_amplitude.csv")
# amp_df = mngs.io.load("./tmp/figs/hist/ripple_amplitude.csv")
# mngs.gen.describe(np.array(amp_df["SWR"]), "median")
# mngs.gen.describe(np.array(amp_df["Control"]), "median")
mngs.io.save(np.log10(amp_df), "./tmp/figs/hist/log10_ripple_amplitude.csv")

