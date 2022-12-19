#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-10 15:39:15 (ywatanabe)"

import sys

sys.path.append(".")
from eeg_ieeg_ripple_clf import utils
import random
import mngs
import pandas as pd
import numpy as np
from utils._load_rips import _determine_firing_patterns


def randomly_define_control_events(rips):
    def randomly_define_a_control_event(rip):
        phase = rip.phase

        phase_dur = PHASE2DUR_DICT[phase]
        phase_start = PHASE2START_SEC[phase]
        phase_end = phase_start + phase_dur

        while True:
            rand_start = phase_start + phase_dur * random.random()
            rand_end = rand_start + rip.duration_ms * 1e-3
            iou = utils.calc_iou([rand_start, rand_end], [rip.start_time, rip.end_time])
            if (iou == 0) * (rand_end <= phase_end):
                return rand_start, rand_end, phase

    starts, ends, phases = [], [], []
    for _, rip in rips.iterrows():
        ss, ee, pp = randomly_define_a_control_event(rip)
        starts.append(ss)
        ends.append(ee)
        phases.append(pp)

    data = np.vstack([starts, ends]).T

    if data.ndim == 1:
        data = data.reshape(1, -1)

    out = pd.DataFrame(
        columns=["start_time", "end_time"],
        data=data,
    )
    out["phase"] = phases # object

    return out


def have_no_overlap(rips, controls):
    ious = []
    for _, rip in rips.iterrows():
        for _, cont in controls.iterrows():
            iou = utils.calc_iou(
                [rip.start_time, rip.end_time], [cont.start_time, cont.end_time]
            )
            ious.append(iou)
    if np.all(np.array(ious) == 0):
        return True
    else:
        return False


def define_controls(trials_uq, rips_df):
    controls = []
    for _, trial in trials_uq.iterrows():
        rips = rips_df[
            (rips_df.subject == trial.subject)
            * (rips_df.session == trial.session)
            * (rips_df.trial_number == trial.trial_number)
        ]
        
        not_overlapped = False
        while not not_overlapped:
            _controls = randomly_define_control_events(rips)
            not_overlapped = have_no_overlap(rips, _controls)

        for k in [
            "subject",
            "session",
            "trial_number",
            "set_size",
            "match",
            "correct",
            "response_time",
            # "probe_letter",
        ]:
            _controls[k] = np.array(rips[k])
        _controls["IoU"] = 0

        controls.append(_controls)

    controls = pd.concat(controls)
    return controls


def load_cons(from_pkl=True, only_correct=True):
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
    rips_df = utils.load_rips(from_pkl=from_pkl, only_correct=only_correct)
    trials_uq = rips_df[["subject", "session", "trial_number"]].drop_duplicates()
    controls = define_controls(trials_uq, rips_df)
    controls["firing_pattern"] = [
        _determine_firing_patterns(controls.iloc[ii]) for ii in range(len(controls))
    ]
    if only_correct:
        controls = controls[controls.correct == True]
    return controls


if __name__ == "__main__":
    controls = load_cons()
