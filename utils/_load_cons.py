#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-03-22 13:39:56 (ywatanabe)"

import sys

sys.path.append(".")
from siEEG_ripple import utils
import random
import mngs
import pandas as pd
import numpy as np
from utils._load_rips import _determine_firing_patterns
from tqdm import tqdm
from ._add_geSWR_grSWR import add_geSWR_grSWR

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


def define_cons(trials_uq, rips_df):
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

    print("defining controls...")
    cons = []

    for i_trial, (_, trial) in enumerate(tqdm(trials_uq.iterrows())):  # 1745
        rips = rips_df[
            (rips_df.subject == trial.subject)
            * (rips_df.session == trial.session)
            * (rips_df.trial_number == trial.trial_number)
        ].copy()

        if rips is pd.DataFrame():
            _cons = rips
        else:
            not_overlapped = False
            # not_overlapped = True # fixme
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


def load_cons(
    from_pkl=True, only_correct=True, ROI=None, extracts_firing_patterns=False
):
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

    rips_df = utils.rips.load_rips(
        from_pkl=from_pkl, only_correct=only_correct, ROI=ROI
    )
    trials_uq = rips_df[["subject", "session", "trial_number"]].drop_duplicates()

    cons = define_cons(trials_uq, rips_df)

    if extracts_firing_patterns:
        print("extracting firing patterns...")

        # for ii in range(len(cons)):
        #     print(_determine_firing_patterns(cons.iloc[ii]))
        # ii = 100
        # _determine_firing_patterns(cons.iloc[ii])
        # ii = 3234
        # _determine_firing_patterns(cons.iloc[ii])
        # ii = 3235
        # _determine_firing_patterns(cons.iloc[ii])
        # ii = 3236
        # _determine_firing_patterns(cons.iloc[ii])
        # ii = 3237
        # _determine_firing_patterns(cons.iloc[ii])

        # _determine_firing_patterns(cons.iloc[0])

        # fixme
        try:
            cons["firing_pattern"] = [
                _determine_firing_patterns(cons.iloc[ii])
                for ii in tqdm(range(len(cons)))
            ]  # fixme
            # single positional indexer is out-of-bounds
        except Exception as e:
            print(e)
            import ipdb

            ipdb.set_trace()

    if only_correct:
        cons = cons[cons.correct == True]

    if from_pkl and (ROI is not None):
        try:
            mngs.io.save(cons, f"./tmp/cons_df/common_average_2.0_SD_{ROI}.pkl")
        except Exception as e:
            print(e)

    cons["center_time"] = ((cons["end_time"] + cons["start_time"]) / 2).astype(float)

    cons = add_geSWR_grSWR(cons)
    return cons


# ii = 0
# i_sub_str = cons_df.iloc[ii].subject
# i_session_str = cons_df.iloc[ii].session
# i_trial = int(cons_df.iloc[ii].trial_number - 1)
# iEEG_ROI = cons_df.iloc[ii].ROI


# iEEG, iEEG_common_ave = utils.load_iEEG(
#     i_sub_str, i_session_str, iEEG_ROI, return_common_averaged_signal=True
# )
# iEEG = iEEG[i_trial]


# # if iEEG.shape[1] == 0: # no channels
# #     return pd.DataFrame(columns=["start_time", "end_time"],
# #                         data=np.array([[np.nan, np.nan]]),
# #                         )

#     # try:
#     #     # bandpass filtering
# import torch
# SAMP_RATE_iEEG = 2000
# LOW_HZ = 80
# HIGH_HZ = 140

# iEEG_ripple_band_passed = np.array(
#     mngs.dsp.bandpass(
#         torch.tensor(np.array(iEEG).astype(np.float32)),
#         SAMP_RATE_iEEG,
#         low_hz=LOW_HZ,
#         high_hz=HIGH_HZ,
#     )
# )
# iEEG_ripple_band_passed[]
# # iEEG_ripple_band_passed_common = np.array(
# #     mngs.dsp.bandpass(
# #         torch.tensor(np.array(iEEG_common_ave).astype(np.float32)),
# #         SAMP_RATE_iEEG,
# #         low_hz=LOW_HZ,
# #         high_hz=HIGH_HZ,
# #     )
# # )
#     # except Exception as e:
#     #     print(e)
#     #     import ipdb; ipdb.set_trace()
#     #         # append ripple band filtered iEEG traces

# ripple_band_iEEG_traces = []
# for i_con, con in cons_df.reset_index().iterrows():
#     start_pts = int(con["start_time"] * SAMP_RATE_iEEG)
#     end_pts = int(con["end_time"] * SAMP_RATE_iEEG)
#     ripple_band_iEEG_traces.append(
#         iEEG_ripple_band_passed[i_trial][:, start_pts:end_pts]
#     )
# rip_df["ripple band iEEG trace"] = ripple_band_iEEG_traces

# # ripple peak amplitude
# ripple_peak_amplitude = [
#     np.abs(rbt).max(axis=-1) for rbt in ripple_band_iEEG_traces
# ]
# ripple_band_baseline_sd = iEEG_ripple_band_passed[i_trial].std(axis=-1)
# rip_df["ripple_peak_amplitude_sd"] = [
#     (rpa / ripple_band_baseline_sd).mean() for rpa in ripple_peak_amplitude
# ]

if __name__ == "__main__":
    # rips = utils.rips.load_rips(ROI="AHR")
    cons = load_cons(ROI="AHR", extracts_firing_patterns=True)  # "AHL"
