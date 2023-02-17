#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-10 16:47:30 (ywatanabe)"

import random
import re
import sys
import warnings

import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm

sys.path.append(".")
from siEEG_ripple import utils



def calc_sim(rips_df, trials_df, suffix=None):
    def _in_calc_sim(pattern_1, pattern_2):
        does_no_zero_patterns_exist = (pattern_1 == 0).all() + (pattern_2 == 0).all()
        if does_no_zero_patterns_exist:
            return np.nan
        else:
            sim = 1 - scipy.spatial.distance.cosine(pattern_1, pattern_2)
            return sim
    
    sim_df = pd.DataFrame()
    for sub in rips_df.subject.unique():
        _rips_df_sub = rips_df[rips_df.subject == sub]

        for session in tqdm(_rips_df_sub.session.unique()):
            trials_info = mngs.io.load(
                f"./data/Sub_{int(sub):02d}/Session_{int(session):02d}/trials_info.csv"
            )

            _rips_df_session = _rips_df_sub[_rips_df_sub.session == session]

            for i_rip_1, i_rip_2 in combinations(np.arange(len(_rips_df_session)), 2):

                rip_1 = _rips_df_session.iloc[i_rip_1]
                rip_2 = _rips_df_session.iloc[i_rip_2]

                probe_letter_1 = trials_info.iloc[
                    int(rip_1.trial_number - 1)
                ].probe_letter

                probe_letter_2 = trials_info.iloc[
                    int(rip_2.trial_number - 1)
                ].probe_letter
                
                pattern_1 = rip_1.firing_pattern
                pattern_2 = rip_2.firing_pattern

                sim = _in_calc_sim(pattern_1, pattern_2)

                try:
                    sim_df_tmp = pd.DataFrame(
                        pd.Series(
                            dict(
                                subject=sub,
                                session=session,
                                i_rip_1=i_rip_1,
                                i_rip_2=i_rip_2,
                                phase_1=rip_1.phase,
                                phase_2=rip_2.phase,
                                probe_letter_1=probe_letter_1,
                                probe_letter_2=probe_letter_2,
                                trial_number_1=rip_1.trial_number,
                                trial_number_2=rip_2.trial_number,
                                similarity=sim,
                                firing_pattern_1=pattern_1,
                                firing_pattern_2=pattern_2,
                                rip_1_dur_ms=(rip_1.end_time - rip_1.start_time) * 1000,
                                rip_2_dur_ms=(rip_2.end_time - rip_2.start_time) * 1000,
                                correct_1=rip_1.correct,
                                correct_2=rip_2.correct,                            
                                IoU_1=rip_1.IoU,
                                IoU_2=rip_2.IoU,                            
                            )
                        )
                    ).T
                except Exception as e:
                    print(e)
                    import ipdb; ipdb.set_trace()

                sim_df = pd.concat([sim_df, sim_df_tmp])


            mngs.io.save(sim_df, f"./tmp/sim_df_{suffix}.pkl")

if __name__ == "__main__":
    from itertools import combinations

    rips_df = utils.load_rips(from_pkl=False, only_correct=False)
    cons_df = utils.load_cons(from_pkl=False, only_correct=False)
    
    trials_df = utils.load_trials(add_n_ripples=True)

    calc_sim(cons_df, trials_df, suffix="cons")
    calc_sim(rips_df, trials_df, suffix="rips")    
