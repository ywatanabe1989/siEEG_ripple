#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-25 15:03:12 (ywatanabe)"

from glob import glob
import mngs
import pandas as pd
import sys
sys.path.append(".")
from siEEG_ripple import utils

ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")
PHASES = mngs.io.load("./config/global.yaml")["PHASES"]
DURS_OF_PHASES = mngs.io.load("./config/global.yaml")["DURS_OF_PHASES"]
SESSION_THRES = mngs.io.load("./config/global.yaml")["SESSION_THRES"]


def load_trials(subs=list(ROIs.keys()), add_n_ripples=False, from_pkl=False, only_correct=False):
    if from_pkl:
        return mngs.io.load("./tmp/trials.pkl")
    
    dfs = []
    for sub in subs:
        sub = f"{int(sub):02d}"
        sessions = [dir[-2:] for dir in glob(f"./data/Sub_{sub}/Session_??")]
        for session in sessions:
            trials_info = mngs.io.load(
                f"./data/Sub_{sub}/Session_{session}/trials_info.csv"
            )
            trials_info["subject"] = sub
            trials_info["session"] = session
            dfs.append(trials_info)
    dfs = pd.concat(dfs).reset_index()
    del dfs["index"], dfs["Unnamed: 0"]

    if add_n_ripples:
        rips_df = utils.load_rips()
        dfs = _add_n_ripples(dfs, rips_df)

    if only_correct:
        # only correct trials
        dfs = dfs[dfs.correct == True]

    # sessions
    dfs = dfs[dfs.session.astype(int) <= SESSION_THRES]

    # add inci
    for phase, dur_sec in zip(PHASES, DURS_OF_PHASES):
        dfs[f"inci_rips_hz_{phase}"] = dfs[f"n_rips_{phase}"] / dur_sec
    
    return dfs

def _add_n_ripples(trials_df, rips_df):
    # phases = ["Fixation", "Encoding", "Maintenance", "Retrieval"]
    for phase in PHASES:
        trials_df[f"n_rips_{phase}"] = 0

    for i_trial, trial in trials_df.iterrows():
        for phase in PHASES:
            indi_subject = rips_df.subject.astype(int) == int(trial.subject)
            indi_session = rips_df.session.astype(int) == int(trial.session)
            indi_trial_number = rips_df.trial_number == trial.trial_number # here
            indi_phase = rips_df.phase == phase
            
            rips = rips_df[indi_subject * indi_session * indi_trial_number * indi_phase]
                
            trials_df[f"n_rips_{phase}"].iloc[i_trial] = len(rips)            

    return trials_df

if __name__ == "__main__":
    # trials_df = load_trials(add_n_ripples=True, from_pkl=False)
    trials_df = load_trials(add_n_ripples=True)    
    mngs.io.save(trials_df, "./tmp/trials.pkl")    

