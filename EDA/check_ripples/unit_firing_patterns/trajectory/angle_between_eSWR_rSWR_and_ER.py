#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-03-21 17:00:04 (ywatanabe)"

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-03-19 13:43:13 (ywatanabe)"
import sys

sys.path.append(".")
import utils
import scipy
from copy import deepcopy
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns

# Functions
def load_gE_gR(subject, session, roi):
    traj = mngs.io.load(
        f"./data/Sub_{subject}/Session_{session}/traj_z_by_session_{roi}.npy"
    )
    traj = traj.transpose(1, 0, 2)
    traj_E = traj[:, :, GS_BINS_DICT["Encoding"][0] : GS_BINS_DICT["Encoding"][1]]
    traj_R = traj[:, :, GS_BINS_DICT["Retrieval"][0] : GS_BINS_DICT["Retrieval"][1]]
    gE = np.nanmedian(traj_E.reshape(len(traj_E), -1), axis=-1)
    gR = np.nanmedian(traj_R.reshape(len(traj_R), -1), axis=-1)
    return gE, gR




def calc_rads(rips_df):
    def _calc_rads_session(rips_df, subject, session, roi):

        gE, gR = load_gE_gR(subject, session, roi)
        v_ER = gR - gE

        rips_df_s = rips_df[(rips_df.subject == subject) * (rips_df.session == session)]
        rips_df_sE = rips_df_s[rips_df_s.phase == "Encoding"]
        rips_df_sR = rips_df_s[rips_df_s.phase == "Retrieval"]

        # SWR_BINS["mid"] = [-2,3] # fixme
        # SWR_BINS["mid"] = [-2,2] # fixme
        v_rips_E = (
            rips_df_sE[f"{SWR_BINS['mid'][1]}"] - rips_df_sE[f"{SWR_BINS['mid'][0]}"]
        )
        v_rips_R = (
            rips_df_sR[f"{SWR_BINS['mid'][1]}"] - rips_df_sR[f"{SWR_BINS['mid'][0]}"]
        )

        rads_ER_eSWR = np.arccos(
            [mngs.linalg.cosine(v_ER, v_rip_E) for v_rip_E in v_rips_E]
        )
        rads_ER_rSWR = np.arccos(
            [mngs.linalg.cosine(v_ER, v_rip_R) for v_rip_R in v_rips_R]
        )

        rads_eSWR_rSWR = []
        for v_rip_E in v_rips_E:
            for v_rip_R in v_rips_R:
                rads_eSWR_rSWR.append(np.arccos(mngs.linalg.cosine(v_rip_E, v_rip_R)))
        rads_eSWR_rSWR = np.array(rads_eSWR_rSWR)

        return rads_ER_eSWR, rads_ER_rSWR, rads_eSWR_rSWR

    rads_ER_eSWR, rads_ER_rSWR, rads_eSWR_rSWR = [], [], []
    for subject, roi in ROIs.items():
        subject = f"{subject:02d}"
        for session in ["01", "02"]:
            _rads_ER_eSWR, _rads_ER_rSWR, _rads_eSWR_rSWR = _calc_rads_session(
                rips_df, subject, session, roi
            )
            rads_ER_eSWR.append(_rads_ER_eSWR)
            rads_ER_rSWR.append(_rads_ER_rSWR)
            rads_eSWR_rSWR.append(_rads_eSWR_rSWR)
    rads_ER_eSWR = np.hstack(rads_ER_eSWR)  # [~np.isnan(rads_ER_eSWR)]
    rads_ER_rSWR = np.hstack(rads_ER_rSWR)
    rads_eSWR_rSWR = np.hstack(rads_eSWR_rSWR)
    return rads_ER_eSWR, rads_ER_rSWR, rads_eSWR_rSWR

def main():
    for match in [1, 2]:
        for set_size in [4,6,8]: # 4, 6, 
            rads_ER_eSWR, rads_ER_rSWR, rads_eSWR_rSWR = calc_rads(
                rips_df[(rips_df.match == match) * (rips_df.set_size == set_size)]
            )

            for rads_str, rads in zip(
                ["ER-eSWR", "ER-rSWR", "eSWR-rSWR"],
                [rads_ER_eSWR, rads_ER_rSWR, rads_eSWR_rSWR],
            ):
            # for rads_str, rads in zip(
            #     ["eSWR-rSWR"],
            #     [rads_eSWR_rSWR],
            # ):
                fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
                mngs.plt.ax_circular_hist(ax, rads)
                ax.set_title(f"{rads_str} match {match} set size {set_size}")
    plt.show()

if __name__ == "__main__":
    import mngs
    import numpy as np

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
    rips_df = utils.rips.add_coordinates(
        utils.rips.load_rips_df_with_traj(BIN_SIZE, is_control=False)
    )
    # cons_df = utils.rips.add_coordinates(
    #     utils.rips.load_rips_df_with_traj(BIN_SIZE, is_control=True)
    # )



    main()
