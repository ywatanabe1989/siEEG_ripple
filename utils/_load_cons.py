#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-23 17:13:31 (ywatanabe)"

import sys

sys.path.append(".")
from siEEG_ripple import utils
import random
import mngs
import pandas as pd
import numpy as np
from utils._load_rips import _determine_firing_patterns
from tqdm import tqdm


def randomly_define_control_events(rips):
    def randomly_define_a_control_event(rip):
        phase = rip.phase

        if phase is None:
            return None, None, None

        phase_dur = PHASE2DUR_DICT[phase]
        phase_start = PHASE2START_SEC[phase]
        phase_end = phase_start + phase_dur

        count = 0
        dur_ms = rip.duration_ms        
        while True:
            
            rand_start = phase_start + phase_dur * random.random()
            rand_end = rand_start + dur_ms * 1e-3
            iou = utils.calc_iou([rand_start, rand_end], [rip.start_time, rip.end_time])

            if (iou == 0) * (rand_end <= phase_end):
                return rand_start, rand_end, phase
            
            if count > 100:
                return rand_start, rand_end, phase
                
            count += 1

    starts, ends, phases = [], [], []
    for _, rip in rips.iterrows():
        ss, ee, pp = randomly_define_a_control_event(rip)
        starts.append(ss)
        ends.append(ee)
        phases.append(pp)

    data = np.vstack([starts, ends]).T

    if data.ndim == 1:
        data = data.reshape(1, -1)

    cons = pd.DataFrame(
        columns=["start_time", "end_time"],
        data=data,
    )
    cons["phase"] = phases  # object

    return cons


def have_no_overlap(rips, cons):
    ious = []
    for _, rip in rips.iterrows():
        for _, cont in cons.iterrows():
            iou = utils.calc_iou(
                [rip.start_time, rip.end_time], [cont.start_time, cont.end_time]
            )
            ious.append(iou)
    if np.all(np.array(ious) == 0):
        return True
    else:
        return False


def define_cons(trials_uq, rips_df):
    print("defining controls...")
    cons = []

    for i_trial, (_, trial) in enumerate(tqdm(trials_uq.iterrows())): # 1745
        rips = rips_df[
            (rips_df.subject == trial.subject)
            * (rips_df.session == trial.session)
            * (rips_df.trial_number == trial.trial_number)
        ].copy()

        if rips is pd.DataFrame():
            _cons = rips
        else:
            not_overlapped = False
            count = 0
            while not not_overlapped:
                _cons = randomly_define_control_events(rips)
                not_overlapped = have_no_overlap(rips, _cons)

                if count > 100:
                    # print(count)                
                    not_overlapped = True
                count += 1

            for k in [
                "subject",
                "session",
                "trial_number",
                "set_size",
                "match",
                "correct",
                "response_time",
                "ROI",
                # "probe_letter",
            ]:
                _cons[k] = np.array(rips[k])
            _cons["IoU"] = 0

        cons.append(_cons)

    cons = pd.concat(cons)
    return cons


def load_cons(from_pkl=True, only_correct=True, ROI=None):
    mngs.general.fix_seeds(random=random)

    global PHASE2DUR_DICT, PHASE2START_SEC
    PHASES = mngs.io.load("./config/global.yaml")["PHASES"]
    DURS_OF_PHASES = mngs.io.load("./config/global.yaml")["DURS_OF_PHASES"]
    PHASE2DUR_DICT = {phase: dur for phase, dur in zip(PHASES, DURS_OF_PHASES)}
    PHASE2START_SEC = {
        "Fixation": 0,
        "Encoding": 1,
        "Maintenance": 3,
        "Retrieval": 6,
    }
    # if from_pkl and (ROI is not None):
    #     try:
    #         return mngs.io.load(f"./tmp/cons_df/common_average_2.0_SD_{ROI}.pkl")
    #     except Exception as e:
    #         print(e)

    rips_df = utils.load_rips(from_pkl=from_pkl, only_correct=only_correct, ROI=ROI)
    trials_uq = rips_df[["subject", "session", "trial_number"]].drop_duplicates()

    cons = define_cons(trials_uq, rips_df)
    print("extracting firing patterns...")
    # 3338
    # single positional indexer is out-of-bound
    # print(cons.iloc[3337])
    # print(cons.iloc[3338])
    # print(cons.iloc[3339])        

    _determine_firing_patterns(cons.iloc[0])    
    
    try:
        cons["firing_pattern"] = [
            _determine_firing_patterns(cons.iloc[ii]) for ii in tqdm(range(len(cons)))
        ] # fixme
        # single positional indexer is out-of-bounds
    except Exception as e:
        print(e)
        import ipdb; ipdb.set_trace()
    if only_correct:
        cons = cons[cons.correct == True]

    if from_pkl and (ROI is not None):
        try:
            mngs.io.save(cons, f"./tmp/cons_df/common_average_2.0_SD_{ROI}.pkl")
        except Exception as e:
            print(e)
        
    return cons


if __name__ == "__main__":
    cons = load_cons(ROI="AHR") # "AHL"
