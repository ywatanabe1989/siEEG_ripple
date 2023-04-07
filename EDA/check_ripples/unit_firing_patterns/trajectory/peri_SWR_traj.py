#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-03-21 13:51:37 (ywatanabe)"
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


def get_ER_based_coords(events_df, subject, session, roi, tgt_bin, adjust_ER_to_O=False):
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
    for ii, (er, ep, rp) in enumerate(zip(ER, EP, RP)):
        try:
            O, A, _B = mngs.linalg.three_line_lengths_to_coords(er, ep, rp)
            B = list(deepcopy(_B))
            # print(f"ER distance: {A[0]}")
            # B[0] /= A[0] # fixme
            if adjust_ER_to_O:
                B[0] -= A[0]/2 # fixme
            ER_based_coords.append(B[:2])
        except Exception as e:
            ER_based_coords.append((np.nan, np.nan))  # , 0

    try:
        ER_based_coords = np.vstack(ER_based_coords).astype(float)
    except Exception as e:
        # print(e)
        ER_based_coords = np.array([np.nan, np.nan]).reshape(1, -1)
    return ER_based_coords


def get_SWR_vectors_based_on_gE_and_gR(rips_df):
    vectors = []
    for subject, roi in ROIs.items():
        subject = f"{subject:02d}"
        for session in ["01", "02"]:
            starts = get_ER_based_coords(rips_df, subject, session, roi, -2, adjust_ER_to_O=False)
            ends = get_ER_based_coords(rips_df, subject, session, roi, 2, adjust_ER_to_O=False)
            vectors.append(ends - starts)
    return np.vstack(vectors)


def get_ER_based_coords_all(event_df, period):
    SWR_BINS = mngs.io.load("./config/global.yaml")["SWR_BINS"]
    start_bin, end_bin = SWR_BINS[period]
    ER_based_coords_all = []
    for tgt_bin in np.arange(start_bin, end_bin):
        ER_based_coords_bin = []
        for subject, roi in ROIs.items():
            subject = f"{subject:02d}"
            for session in ["01", "02"]:
                ER_based_coords_bin.append(
                    get_ER_based_coords(
                        event_df,
                        subject,
                        session,
                        roi,
                        tgt_bin,
                        adjust_ER_to_O=True,
                    )
                )
        ER_based_coords_all.append(np.vstack(ER_based_coords_bin))
    ER_based_coords_all = np.stack(ER_based_coords_all, axis=0)
    ER_based_coords_all = np.nanmean(ER_based_coords_all, axis=0)
    # ER_based_coords_all[:, 0] -= 0.5
    ER_based_coords_all = pd.DataFrame(
        ER_based_coords_all, columns=[f"{period}_x", f"{period}_y"]
    )
    return ER_based_coords_all


def plot_scatter(ER_based_coords):
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ss = 1
    for ii in range(len(ER_based_coords)):
        ax.scatter(
            ER_based_coords["pre_x"] - .5,
            ER_based_coords["pre_y"],
            # color="white",
            # edgecolor="black",
            color=mngs.plt.colors.to_RGBA(
                "gray"
            ),  # "black",            #            color="black",
            s=ss,
            marker="x",
            alpha=0.5,
        )
        ax.scatter(
            ER_based_coords["mid_x"] - .5,
            ER_based_coords["mid_y"],
            # color="darkgray",
            color=mngs.plt.colors.to_RGBA("purple"),  # "black",
            s=ss,
            marker="x",
            alpha=0.5,
        )
        ax.scatter(
            ER_based_coords["post_x"] - .5,
            ER_based_coords["post_y"],
            color="black",
            s=ss,
            marker="x",
            alpha=0.5,
        )
        # cols = ["pre_x", "pre_y", "mid_x", "mid_y", "post_x", "post_y"]
        # for pattern in [1,2]:
        #     if pattern == 1:
        #         x1, y1, x2, y2 = cols[:4]
        #     if pattern == 2:
        #         x1, y1, x2, y2 = cols[2:]
        #     ax.arrow(
        #         ER_based_coords[x1].iloc[ii],
        #         ER_based_coords[y1].iloc[ii],
        #         ER_based_coords[x2].iloc[ii]
        #         - ER_based_coords[x1].iloc[ii],
        #         ER_based_coords[y2].iloc[ii]
        #         - ER_based_coords[y1].iloc[ii],
        #         # head_width=0.01,
        #         linestyle="dotted",
        #         color="lightgray",
        #         # width=0.05,
        #     )
    xlim = 2.5
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(0, 5)
    return fig


def to_df(ER_based_coords, match, set_size, swr_phase):

    dfs = []
    for i_period, period in enumerate(["pre", "mid", "post"]):
        dists_from_E = [
            np.linalg.norm(
                np.array((0.0, 0))
                - np.array(
                    (
                        ER_based_coords[f"{period}_x"].iloc[ii],
                        ER_based_coords[f"{period}_y"].iloc[ii],
                    )
                ),
            )
            for ii in range(len(ER_based_coords))
        ]
        df_E = pd.DataFrame({"dist": dists_from_E})
        df_E["from"] = "Encoding"

        dists_from_R = [
            np.linalg.norm(
                np.array((0.0, 1))
                - np.array(
                    (
                        ER_based_coords[f"{period}_x"].iloc[ii],
                        ER_based_coords[f"{period}_y"].iloc[ii],
                    )
                ),
            )
            for ii in range(len(ER_based_coords))
        ]
        df_R = pd.DataFrame({"dist": dists_from_R})
        df_R["from"] = "Retrieval"

        # df_R = pd.DataFrame(dists_from_R)
        # df_R["SWR phase"] = "Retrieval"]

        # dists[f"from_R"] = dists_from_R
        df = pd.concat([df_E, df_R])
        # df["SWR_phase"] = phase  # fixme
        df["match"] = match
        df["set_size"] = set_size
        df["period"] = period
        df["SWR phase"] = swr_phase
        dfs.append(df)

    return pd.concat(dfs)


# def circular_hist(
#     ax, x, bins=16, density=True, offset=0, gaps=True, color=None, range_bias=0
# ):
#     """
#     Produce a circular histogram of angles on ax.

#     Parameters
#     ----------
#     ax : matplotlib.axes._subplots.PolarAxesSubplot
#         axis instance created with subplot_kw=dict(projection='polar').

#     x : array
#         Angles to plot, expected in units of radians.

#     bins : int, optional
#         Defines the number of equal-width bins in the range. The default is 16.

#     density : bool, optional
#         If True plot frequency proportional to area. If False plot frequency
#         proportional to radius. The default is True.

#     offset : float, optional
#         Sets the offset for the location of the 0 direction in units of
#         radians. The default is 0.

#     gaps : bool, optional
#         Whether to allow gaps between bins. When gaps = False the bins are
#         forced to partition the entire [-pi, pi] range. The default is True.

#     Returns
#     -------
#     n : array or list of arrays
#         The number of values in each bin.

#     bins : array
#         The edges of the bins.

#     patches : `.BarContainer` or list of a single `.Polygon`
#         Container of individual artists used to create the histogram
#         or list of such containers if there are multiple input datasets.
#     """
#     # Wrap angles to [-pi, pi)
#     x = (x + np.pi) % (2 * np.pi) - np.pi

#     # Force bins to partition entire circle
#     if not gaps:
#         bins = np.linspace(-np.pi, np.pi, num=bins + 1)

#     # Bin data and record counts
#     n, bins = np.histogram(
#         x, bins=bins, range=(-np.pi + range_bias, np.pi + range_bias)
#     )

#     # Compute width of each bin
#     widths = np.diff(bins)

#     # By default plot frequency proportional to area
#     if density:
#         # Area to assign each bin
#         area = n / x.size
#         # Calculate corresponding bin radius
#         radius = (area / np.pi) ** 0.5
#     # Otherwise plot frequency proportional to radius
#     else:
#         radius = n

#     # fixme
#     # med_val = np.pi/2#
#     med_val = np.nanmedian(x)
#     ax.axvline(med_val, color=color)

#     # Plot data on ax
#     patches = ax.bar(
#         bins[:-1],
#         radius,
#         zorder=1,
#         align="edge",
#         width=widths,
#         # edgecolor="C0",
#         edgecolor=color,
#         alpha=0.9,
#         fill=False,
#         linewidth=1,
#     )

#     # Set the direction of the zero angle
#     ax.set_theta_offset(offset)

#     # Remove ylabels for area plots (they are mostly obstructive)
#     if density:
#         ax.set_yticks([])

#     return n, bins, patches


def plot_polar(ax, SWR_directions, set_size):
    color = {4: "light_blue", 6: "purple", 8: "navy"}[set_size]
    color = mngs.plt.colors.to_RGBA(color, alpha=0.9)
    rads = []
    for ii, SWR_direction in enumerate(SWR_directions):
        hypo = np.sqrt(SWR_direction[0] ** 2 + SWR_direction[1] ** 2)
        cos = SWR_direction[0] / hypo
        rads.append(np.arccos(cos))
    rads = np.array(rads)
    rads = rads[~np.isnan(rads)]
    mngs.plt.circular_hist(ax, rads, color=color, range_bias=i_period * 5 / 360 * np.pi)
    return ax


# def plot_polar(ax, _ER_based_coords):
#     for i_period, period in enumerate(["pre", "mid", "post"]):
#         color = mngs.plt.colors.to_RGBA(["light_blue", "purple", "navy"][i_period], alpha=0.9)
#         rads = []
#         for ii in range(len(_ER_based_coords)):
#             ER = np.sqrt(
#                 _ER_based_coords.iloc[ii][f"{period}_y"] ** 2
#                 + _ER_based_coords.iloc[ii][f"{period}_x"] ** 2
#             )
#             cos = _ER_based_coords.iloc[ii][f"{period}_x"] / ER
#             rads.append(np.arccos(cos))
#         rads = np.array(rads)
#         rads = rads[~np.isnan(rads)]
#         circular_hist(ax, rads, color=color, range_bias=i_period*5/360*np.pi)
#     return ax


def plot_dist(df):
    # Distance
    fig, axes = plt.subplots(ncols=4, sharex=True, sharey=True)
    for i_match, match in enumerate(df.match.unique()):
        for i_SWR_phase, SWR_phase in enumerate(df["SWR phase"].unique()):
            # for i_from, _from in enumerate(df["from"].unique()):
            _from = ["Encoding", "Retrieval"][0]
            ax = axes[2 * i_match + i_SWR_phase]
            sns.boxplot(
                data=df[
                    (df.match == match)
                    * (df["from"] == "Encoding")
                    * (df["SWR phase"] == SWR_phase)
                ],
                # x=xx,
                x="period",
                # hue="set_size",
                y="dist",
                showfliers=False,
                ax=ax,
            )
            ax.set_title(
                f"match: {match}\n{SWR_phase[0].lower()}SWR from g{_from[0]}\n"
            )
    return fig


def main():
    dfs = []
    for i_swr_phase, swr_phase in enumerate(["Encoding", "Retrieval"]):
        for match in [1, 2]:
            fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
            for set_size in [4, 6, 8]:
                rips_df_msp = rips_df[
                    (rips_df.match == match)
                    * (rips_df.set_size == set_size)
                    * (rips_df.phase == swr_phase)
                    * (rips_df.correct == True)
                ]

                ER_based_coords = []
                for i_period, period in enumerate(["pre", "mid", "post"]):
                    _ER_based_coords = get_ER_based_coords_all(
                        rips_df_msp,
                        period,
                    )
                    ER_based_coords.append(_ER_based_coords)
                ER_based_coords = pd.concat(ER_based_coords, axis=1)

                SWR_directions = get_SWR_vectors_based_on_gE_and_gR(rips_df_msp)
                # polar, angle
                ax = plot_polar(ax, SWR_directions, set_size)
                # ax = plot_polar(ax, ER_based_coords)
                mngs.io.save(
                    fig,
                    f"./tmp/figs/polar/SWR_directions_based_on_gE_and_gR/match_{match}/"
                    f"{i_swr_phase}_{swr_phase[0].lower()}SWR.tif",
                )

                # scatter, positions
                fig = plot_scatter(ER_based_coords)
                mngs.io.save(
                    fig,
                    f"./tmp/figs/scatter/peri_SWR_pos_around_gE_and_gR/match_{match}/"
                    f"{i_swr_phase}_{swr_phase}_{set_size}.tif",
                )

                # buffering
                dfs.append(to_df(ER_based_coords, match, set_size, swr_phase))
                # dfs.append(to_df(ER_based_coords, swr_phase))

    # box, dist
    mngs.io.save(
        plot_dist(pd.concat(dfs)),
        f"./tmp/figs/box/peri_SWR_pos_around_gE_and_gR/dist.tiff",
    )

    # plt.show()

    # E_rate = round(np.mean([pp[0] < 0 for pp in ER_based_coords]), 3)
    # fig = plot(ER_based_coords)
    # mngs.io.save(
    #     fig,
    #     f"./tmp/figs/scatter/peri_SWR_pos_around_gE_and_gR/match_{match}/"
    #     f"{i_phase}_{phase}/{i_period}_{period}_{E_rate}.png",
    # )


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
    # cons_df = utils.rips.add_coordinates(
    #     utils.rips.load_rips_df_with_traj(BIN_SIZE, is_control=True)
    # )

    main()
