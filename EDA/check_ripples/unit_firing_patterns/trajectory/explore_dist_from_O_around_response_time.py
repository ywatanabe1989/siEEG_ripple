#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-03-21 15:21:38 (ywatanabe)"
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def collect_peri_response_time_distance_from_O():
    def _collect_peri_response_time_distance_from_O_session(subject, session, roi):
        traj = mngs.io.load(
            f"./data/Sub_{subject}/Session_{session}/traj_z_by_session_{roi}.npy"
        )
        trials_info = mngs.io.load(
            f"./data/Sub_{subject}/Session_{session}/trials_info.csv"
        )

        # fixme
        # indi = (trials_info.set_size == 8) * (trials_info.match == 1)
        # indi = trials_info.match == 1
        # traj = traj[indi]
        # trials_info = trials_info[indi]

        # dist_from_O = np.linalg.norm(traj, axis=1)
        # trials_info["rt_bin"] = (
        #     (trials_info["response_time"] + 6) / (50 / 1000)
        # ).astype(int)
        # trials_info["rt_bin"] = 140
        trials_info["rt_bin"] = 40

        dists = mngs.gen.listed_dict()
        for delta_bin in range(-20, 21):
            for _, trial in trials_info.iterrows():
                try:
                    dd = dist_from_O[
                        int(trial.trial_number - 1), trial.rt_bin + delta_bin
                    ]
                except Exception as e:
                    print(e)
                    dd = np.nan
                dists[delta_bin].append(dd)

        dists = pd.DataFrame(dists)
        return dists

    dists = []
    for subject, roi in ROIs.items():
        subject = f"{subject:02d}"
        for session in ["01", "02"]:
            dists.append(
                _collect_peri_response_time_distance_from_O_session(
                    subject, session, roi
                )
            )
    return pd.concat(dists)


if __name__ == "__main__":
    import mngs
    import numpy as np
    import sys

    sys.path.append(".")
    from siEEG_ripple import utils

    # Fixes seeds
    mngs.gen.fix_seeds(42, np=np)

    # Parameters
    (
        PHASES,
        PHASES_BINS_DICT,
        GS_BINS_DICT,
        COLORS_DICT,
        BIN_SIZE,
    ) = utils.define_phase_time()

    SWR_BINS = mngs.io.load("./config/global.yaml")["SWR_BINS"]

    ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")

    dist_df = collect_peri_response_time_distance_from_O()

    mm = dist_df.apply(np.mean)
    ss = dist_df.apply(np.std)

    fig, ax = plt.subplots()
    ax = mngs.plt.ax_fill_between(ax, np.arange(len(mm)), mm, ss, None)
    plt.show()
