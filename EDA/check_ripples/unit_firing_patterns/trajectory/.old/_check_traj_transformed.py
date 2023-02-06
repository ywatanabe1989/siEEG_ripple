#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-01-12 18:50:01 (ywatanabe)"

from glob import glob
import mngs
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def main(roi):
    LPATHs = glob(f"data/Sub_0?/Session_0?/traj_transformed_{roi}.npy")

    trajs = np.stack(
        [np.nanmedian(mngs.io.load(lpath), axis=0) for lpath in LPATHs], axis=0
    )
    # trajs = np.vstack([mngs.io.load(lpath) for lpath in LPATHs])

    n_dims = 3
    fig, axes = plt.subplots(nrows=n_dims, sharex=True, sharey=True)
    for i_dim, ax in enumerate(axes):
        data = trajs[:, i_dim, :].T
        # data += 5000
        ax.plot(data)  # +1e5)

        # ax.set_ylim(-5000, 5000)
        ax.set_yscale("log")
    fig.suptitle(roi)
    axes[0].set_ylabel("Encoding")
    axes[1].set_ylabel("Maintenance")
    axes[2].set_ylabel("Retrieval")
    return fig, trajs
    # plt.show()


if __name__ == "__main__":
    trajs_all = []
    ROIs = ["AHL", "AHR", "PHL", "PHR", "ECL", "ECR", "AL", "AR"]
    for roi in ROIs:
        fig, trajs = main(roi)
        trajs_all.append(trajs)
        # plt.show()
        mngs.io.save(fig, f"./tmp/figs/line/neural_trajectory_transformed/{roi}.png")
        plt.close()
    # main("AHR")
    # main("PHR")
    # main("PHL")
    # main("ECL")
    # main("ECR")
    # main("AL")
    # main("AR")

    for trajs in trajs_all:
        print(trajs.shape)

    trajs_all = np.stack([np.nanmedian(trajs, axis=0) for trajs in trajs_all], axis=0)

    n_phases = 3
    fig, axes = plt.subplots(nrows=n_phases, sharey=True)
    for i_phase in range(n_phases):
        ax = axes[i_phase]
        for i_roi, _ in enumerate(ROIs):
            ax.plot(trajs_all[i_roi, i_phase, :].T, label=ROIs[i_roi])
        ax.set_ylabel(["Encoding", "Maintenance", "Retrieval"][i_phase])
        ax.legend()

    plt.show()
