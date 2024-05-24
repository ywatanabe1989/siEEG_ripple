#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-04-18 20:34:10 (ywatanabe)"

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-03-19 13:43:13 (ywatanabe)"
import sys

sys.path.append(".")
import utils
import scipy
from copy import deepcopy
import matplotlib

matplotlib.use("Agg")
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


def get_ER_based_coords(events_df, subject, session, roi, tgt_bin):
    gE, gR = load_gE_gR(subject, session, roi)

    events_df_s = events_df[
        (events_df.subject == subject) * (events_df.session == session)
    ]

    ER = [np.linalg.norm(gE - gR) for _ in range(len(events_df_s))]  # ER
    EP = [
        np.linalg.norm(events_df_s[f"{tgt_bin}"].iloc[ii] - gE)
        for ii in range(len(events_df_s))
    ]
    RP = [
        np.linalg.norm(events_df_s[f"{tgt_bin}"].iloc[ii] - gR)
        for ii in range(len(events_df_s))
    ]

    ER_based_coords = []
    R_coords = []
    for ii, (er, ep, rp) in enumerate(zip(ER, EP, RP)):
        try:
            O, A, _B = mngs.linalg.three_line_lengths_to_coords(er, ep, rp)
            B = list(deepcopy(_B))
            R_coords.append(A)
            # print(f"ER distance: {A[0]}")
            # B[0] /= A[0] # fixme
            # if adjust_ER_to_O:
            #     B[0] -= A[0] / 2  # fixme
            ER_based_coords.append(B[:2])
        except Exception as e:
            ER_based_coords.append((np.nan, np.nan))  # , 0
            R_coords.append((np.nan, np.nan, np.nan))

    try:
        ER_based_coords = np.vstack(ER_based_coords).astype(float)
        R_coords = np.vstack(R_coords).astype(float)
    except Exception as e:
        # print(e)
        ER_based_coords = np.array([np.nan, np.nan]).reshape(1, -1)
        R_coords = np.array([np.nan, np.nan, np.nan]).reshape(1, -1)
    return ER_based_coords, R_coords


def get_ER_based_coords_all(event_df, period):
    SWR_BINS = mngs.io.load("./config/global.yaml")["SWR_BINS"]
    start_bin, end_bin = SWR_BINS[period]
    ER_based_coords_all = []
    R_coords_all = []
    for tgt_bin in np.arange(start_bin, end_bin):
        ER_based_coords_bin = []
        R_coords_bin = []
        for subject, roi in ROIs.items():
            subject = f"{subject:02d}"
            for session in ["01", "02"]:
                _ER_based_coords_bin, _R_coords_bin = get_ER_based_coords(
                    event_df,
                    subject,
                    session,
                    roi,
                    tgt_bin,
                )
                ER_based_coords_bin.append(_ER_based_coords_bin)
                R_coords_bin.append(_R_coords_bin)
                # ER_based_coords_bin.append(
                #     get_ER_based_coords(
                #         event_df,
                #         subject,
                #         session,
                #         roi,
                #         tgt_bin,
                #     )[0]
                # )
                # R_coords_bin.append(
                #     get_ER_based_coords(
                #         event_df,
                #         subject,
                #         session,
                #         roi,
                #         tgt_bin,
                #     )[1]
                # )
        ER_based_coords_all.append(np.vstack(ER_based_coords_bin))
        R_coords_all.append(np.vstack(R_coords_bin))
    ER_based_coords_all = np.stack(ER_based_coords_all, axis=0)
    ER_based_coords_all = np.nanmean(ER_based_coords_all, axis=0)
    R_coords_all = np.stack(R_coords_all, axis=0)
    R_coords_all = np.nanmean(R_coords_all, axis=0)
    # ER_based_coords_all[:, 0] -= 0.5
    ER_based_coords_all = pd.DataFrame(
        ER_based_coords_all, columns=[f"{period}_x", f"{period}_y"]
    )
    R_coords_all = pd.DataFrame(
        R_coords_all, columns=[f"{period}_x", f"{period}_y", f"{period}_z"]
    )
    return ER_based_coords_all, R_coords_all


def plot_scatter(ER_based_coords):
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ss = 1
    for ii in range(len(ER_based_coords)):
        ax.scatter(
            ER_based_coords["pre_x"],
            ER_based_coords["pre_y"],
            color=mngs.plt.colors.to_RGBA("gray"),
            s=ss,
            marker="x",
            alpha=0.5,
        )
        ax.scatter(
            ER_based_coords["mid_x"],
            ER_based_coords["mid_y"],
            color=mngs.plt.colors.to_RGBA("purple"),
            s=ss,
            marker="x",
            alpha=0.5,
        )
        ax.scatter(
            ER_based_coords["post_x"],
            ER_based_coords["post_y"],
            color="black",
            s=ss,
            marker="x",
            alpha=0.5,
        )
    xlim = 2.5
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(0, 5)
    return fig


def main():
    for event_str, events_df in zip(["SWR-", "SWR+"], [cons_df, rips_df]):
        for i_swr_phase, swr_phase in enumerate(["Encoding", "Retrieval"]):
            for match in [1, 2]:
                fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
                ER_based_coords_all_set_sizes = []
                # R_coords_all_set_sizes = []
                for set_size in [4, 6, 8]:
                    events_df_msp = events_df[
                        (events_df.match == match)
                        * (events_df.set_size == set_size)
                        * (events_df.phase == swr_phase)
                        * (events_df.correct == True)
                    ]

                    ER_based_coords = []
                    R_coords = []
                    for i_period, period in enumerate(["pre", "mid", "post"]):
                        _ER_based_coords, _R_coords = get_ER_based_coords_all(
                            events_df_msp,
                            period,
                        )
                        ER_based_coords.append(_ER_based_coords)
                        R_coords.append(_R_coords)
                    ER_based_coords = pd.concat(ER_based_coords, axis=1)
                    R_coords = pd.concat(R_coords, axis=1)

                    # # adjust the ER center as O
                    # for i_period, period in enumerate(["pre", "mid", "post"]):
                    #     ER_based_coords[f"{period}_x"] -= R_coords[f"{period}_x"] / 2

                    ER_based_coords["R_x"] = R_coords["pre_x"]

                    mngs.io.save(
                        ER_based_coords,
                        f"./tmp/figs/scatter/peri_SWR_pos_around_gE_and_gR_new/{event_str}/ER_based_coords/match_{match}/"
                        f"{i_swr_phase}_{swr_phase}_{set_size}.csv",
                    )

                    # scatter, positions
                    fig = plot_scatter(ER_based_coords)
                    mngs.io.save(
                        fig,
                        f"./tmp/figs/scatter/peri_SWR_pos_around_gE_and_gR_new/images/{event_str}/match_{match}/"
                        f"{i_swr_phase}_{swr_phase}_{set_size}.tif",
                    )

                    # for all set sizes
                    ER_based_coords_all_set_sizes.append(ER_based_coords)
                    # R_coords_all_set_sizes.append(R_coords)

                ER_based_coords_all_set_sizes = pd.concat(ER_based_coords_all_set_sizes)
                # R_coords_all_set_sizes = pd.concat(R_coords_all_set_sizes)
                mngs.io.save(
                    ER_based_coords_all_set_sizes,
                    f"./tmp/figs/scatter/peri_SWR_pos_around_gE_and_gR_new/{event_str}/ER_based_coords/match_{match}/"
                    f"{i_swr_phase}_{swr_phase}_468.csv",
                )


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

    ROIs = mngs.io.load("./config/ripple_detectable_ROI.yaml")
    rips_df = utils.rips.add_coordinates(
        utils.rips.load_rips_df_with_traj(BIN_SIZE, is_control=False)
    )
    cons_df = utils.rips.add_coordinates(
        utils.rips.load_rips_df_with_traj(BIN_SIZE, is_control=True)
    )

    main()
