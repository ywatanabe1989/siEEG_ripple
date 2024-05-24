#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-04-11 15:16:49 (ywatanabe)"

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
    # gE = np.nanmean(traj_E.reshape(len(traj_E), -1), axis=-1) # fixme
    # gR = np.nanmean(traj_R.reshape(len(traj_R), -1), axis=-1) # fixme
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

        # rads_ER_eSWR = np.arccos(
        #     [mngs.linalg.cosine(v_ER, v_rip_E) for v_rip_E in v_rips_E]
        # )
        # rads_ER_rSWR = np.arccos(
        #     [mngs.linalg.cosine(v_ER, v_rip_R) for v_rip_R in v_rips_R]
        # )
        rads_ER_eSWR = np.array(
            [mngs.linalg.cosine(v_ER, v_rip_E) for v_rip_E in v_rips_E]
        )  # fixme
        rads_ER_rSWR = np.array(
            [mngs.linalg.cosine(v_ER, v_rip_R) for v_rip_R in v_rips_R]
        )  # fixme

        rads_eSWR_rSWR = []
        for v_rip_E in v_rips_E:
            for v_rip_R in v_rips_R:
                # rads_eSWR_rSWR.append(np.arccos(mngs.linalg.cosine(v_rip_E, v_rip_R)))
                rads_eSWR_rSWR.append(mngs.linalg.cosine(v_rip_E, v_rip_R))  # fixme
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


from scipy import stats


def main():
    for event_str, events_df in zip(["SWR-", "SWR+"], [cons_df, rips_df]):
        for match in [1, 2]:
            for i_rads, rads_str in enumerate(["ER-eSWR", "ER-rSWR", "eSWR-rSWR"]):
                # fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
                fig, ax = plt.subplots()
                data = []
                colors = []
                xx = np.linspace(-1, 1, 17)
                xx = (xx + (xx[1] - xx[0])/2)[:-1]
                kde_out = pd.DataFrame()
                for set_size in [4, 6, 8]:
                    rads_ER_eSWR, rads_ER_rSWR, rads_eSWR_rSWR = calc_rads(
                        events_df[(events_df.match == match) * (events_df.set_size == set_size)]
                    )
                    rads = np.array([rads_ER_eSWR, rads_ER_rSWR, rads_eSWR_rSWR][i_rads])
                    data.append(rads)

                    color = mngs.plt.colors.to_RGBA(
                        {4: "light_blue", 6: "purple", 8: "navy"}[set_size]
                    )
                    colors.append(color)

                    kde = stats.gaussian_kde(rads[~np.isnan(rads)])

                    ax.plot(xx, kde(xx), color=color)

                    kde_out_new = pd.DataFrame({
                        "xx":xx,                                    
                        "kde":kde(xx),
                        "set_size": set_size,
                        })
                    kde_out = pd.concat([kde_out, kde_out_new])

                    # mngs.plt.ax_circular_hist(ax, rads, color=mngs.plt.colors.to_RGBA(color))
                    # ax.hist(rads, bins=16, range=(0, np.pi), alpha=.5,
                    # color=mngs.plt.colors.to_RGBA(color), density=True)
                    # ax.hist(rads, bins=16, range=(-1, 1), alpha=.5,
                    # color=mngs.plt.colors.to_RGBA(color), density=True)


                out = {ss:dd for ss,dd in zip([4,6,8], data)}
                out["all"] = np.hstack(data)
                out = mngs.gen.force_dataframe(out)
                mngs.io.save(
                    out,
                    f"./tmp/figs/box/eSWR_vs_rSWR_vs_ER_new/{event_str}/match_{match}/{rads_str}.csv",
                )

                ns, bins, patches = ax.hist(
                    data, bins=16, range=(-1, 1), alpha=0.5, color=colors, density=True
                )
                ns_all, bins, patches = ax.hist(
                    np.hstack(data), bins=16, range=(-1, 1), alpha=0.5, color="black", density=True
                )
                kde_all = stats.gaussian_kde(np.hstack(data)[~np.isnan(np.hstack(data))])
                kde_out["kde_all"] = np.hstack([kde_all(xx) for _ in range(3)])
                kde_out["count"] = np.hstack(ns)
                kde_out["count_all"] = np.hstack([ns_all for _ in range(3)])
                mngs.io.save(
                    kde_out,
                    f"./tmp/figs/polar/eSWR_vs_rSWR_vs_ER_new/{event_str}/match_{match}/{rads_str}.csv",
                )

                ax.set_title(f"{rads_str} match {match}")
                mngs.io.save(
                    fig,
                    f"./tmp/figs/polar/eSWR_vs_rSWR_vs_ER_new/{event_str}/match_{match}/{rads_str}.tif",
                )
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
    cons_df = utils.rips.add_coordinates(
        utils.rips.load_rips_df_with_traj(BIN_SIZE, is_control=True)
    )

    main()
