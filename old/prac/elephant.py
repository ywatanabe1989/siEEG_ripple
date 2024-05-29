#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-12-22 18:20:31 (ywatanabe)"

# !pip install elephant

import numpy as np
from scipy.integrate import odeint
import quantities as pq
import neo
from elephant.spike_train_generation import inhomogeneous_poisson_process
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from elephant.gpfa import GPFA


def integrated_oscillator(
    dt, num_steps, x0=0, y0=1, angular_frequency=2 * np.pi * 1e-3
):
    """
    Parameters
    ----------
    dt : float
        Integration time step in ms.
    num_steps : int
        Number of integration steps -> max_time = dt*(num_steps-1).
    x0, y0 : float
        Initial values in three dimensional space.
    angular_frequency : float
        Angular frequency in 1/ms.

    Returns
    -------
    t : (num_steps) np.ndarray
        Array of timepoints
    (2, num_steps) np.ndarray
        Integrated two-dimensional trajectory (x, y, z) of the harmonic oscillator
    """

    assert isinstance(num_steps, int), "num_steps has to be integer"
    t = dt * np.arange(num_steps)
    x = x0 * np.cos(angular_frequency * t) + y0 * np.sin(angular_frequency * t)
    y = -x0 * np.sin(angular_frequency * t) + y0 * np.cos(angular_frequency * t)
    return t, np.array((x, y))


def integrated_lorenz(
    dt, num_steps, x0=0, y0=1, z0=1.05, sigma=10, rho=28, beta=2.667, tau=1e3
):
    """

    Parameters
    ----------
    dt :
        Integration time step in ms.
    num_steps : int
        Number of integration steps -> max_time = dt*(num_steps-1).
    x0, y0, z0 : float
        Initial values in three dimensional space
    sigma, rho, beta : float
        Parameters defining the lorenz attractor
    tau : characteristic timescale in ms

    Returns
    -------
    t : (num_steps) np.ndarray
        Array of timepoints
    (3, num_steps) np.ndarray
        Integrated three-dimensional trajectory (x, y, z) of the Lorenz attractor
    """

    def _lorenz_ode(point_of_interest, timepoint, sigma, rho, beta, tau):
        """
        Fit the model with `spiketrains` data and apply the dimensionality
        reduction on `spiketrains`.

        Parameters
        ----------
        point_of_interest : tuple
            Tupel containing coordinates (x,y,z) in three dimensional space.
        timepoint : a point of interest in time
        dt :
            Integration time step in ms.
        num_steps : int
            Number of integration steps -> max_time = dt*(num_steps-1).
        sigma, rho, beta : float
            Parameters defining the lorenz attractor
        tau : characteristic timescale in ms

        Returns
        -------
        x_dot, y_dot, z_dot : float
            Values of the lorenz attractor's partial derivatives
            at the point x, y, z.
        """

        x, y, z = point_of_interest

        x_dot = (sigma * (y - x)) / tau
        y_dot = (rho * x - y - x * z) / tau
        z_dot = (x * y - beta * z) / tau
        return x_dot, y_dot, z_dot

    assert isinstance(num_steps, int), "num_steps has to be integer"

    t = dt * np.arange(num_steps)
    poi = (x0, y0, z0)
    return t, odeint(_lorenz_ode, poi, t, args=(sigma, rho, beta, tau)).T


def random_projection(data, embedding_dimension, loc=0, scale=None):
    """
    Parameters
    ----------
    data : np.ndarray
        Data to embed, shape=(M, N)
    embedding_dimension : int
        Embedding dimension, dimensionality of the space to project to.
    loc : float or array_like of floats
        Mean (“centre”) of the distribution.
    scale : float or array_like of floats
        Standard deviation (spread or “width”) of the distribution.

    Returns
    -------
    np.ndarray
       Random (normal) projection of input data, shape=(dim, N)

    See Also
    --------
    np.random.normal()

    """
    if scale is None:
        scale = 1 / np.sqrt(data.shape[0])
    projection_matrix = np.random.normal(
        loc, scale, (embedding_dimension, data.shape[0])
    )
    return np.dot(projection_matrix, data)


def generate_spiketrains(instantaneous_rates, num_trials, timestep):
    """
    Parameters
    ----------
    instantaneous_rates : np.ndarray
        Array containing time series.
    timestep :
        Sample period.
    num_steps : int
        Number of timesteps -> max_time = timestep*(num_steps-1).

    Returns
    -------
    spiketrains : list of neo.SpikeTrains
        List containing spiketrains of inhomogeneous Poisson
        processes based on given instantaneous rates.

    """

    spiketrains = []
    for _ in range(num_trials):
        spiketrains_per_trial = []
        for inst_rate in instantaneous_rates:
            anasig_inst_rate = neo.AnalogSignal(
                inst_rate, sampling_rate=1 / timestep, units=pq.Hz
            )
            spiketrains_per_trial.append(
                inhomogeneous_poisson_process(anasig_inst_rate)
            )
        spiketrains.append(spiketrains_per_trial)

    return spiketrains


def plot_oscillator_trajectory_2dim(
    times_oscillator, trajectory_2dim, num_spiketrains, spiketrains_oscillator
):
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    ax1.set_title("2-dim Harmonic Oscillator")
    ax1.set_xlabel("time [s]")
    for i, y in enumerate(oscillator_trajectory_2dim):
        ax1.plot(times_oscillator, y, label=f"dimension {i}")
    ax1.legend()

    ax2.set_title("Trajectory in 2-dim space")
    ax2.set_xlabel("Dim 1")
    ax2.set_ylabel("Dim 2")
    ax2.set_aspect(1)
    ax2.plot(oscillator_trajectory_2dim[0], oscillator_trajectory_2dim[1])

    ax3.set_title(f"Projection to {num_spiketrains}-dim space")
    ax3.set_xlabel("time [s]")
    y_offset = oscillator_trajectory_Ndim.std() * 3
    for i, y in enumerate(oscillator_trajectory_Ndim):
        ax3.plot(times_oscillator, y + i * y_offset)

    trial_to_plot = 0
    ax4.set_title(f"Raster plot of trial {trial_to_plot}")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Spike train index")
    for i, spiketrain in enumerate(spiketrains_oscillator[trial_to_plot]):
        ax4.plot(spiketrain, np.ones_like(spiketrain) * i, ls="", marker="|")

    plt.tight_layout()
    plt.show()


def plot_trajectories(trajectories):
    # plots trajectries
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    color_single_trial = "C0"
    alpha_single_trial = 0.5

    linewidth_trial_average = 2
    color_trial_average = "C1"

    ax1.set_title("Original latent dynamics")
    ax1.set_xlabel("Dim 1")
    ax1.set_ylabel("Dim 2")
    ax1.set_aspect(1)
    ax1.plot(oscillator_trajectory_2dim[0], oscillator_trajectory_2dim[1])

    ax2.set_title("Latent dynamics extracted by GPFA")
    ax2.set_xlabel("Dim 1")
    ax2.set_ylabel("Dim 2")
    ax2.set_aspect(1)
    # single trial trajectories
    for single_trial_trajectory in trajectories:
        ax2.plot(
            single_trial_trajectory[0],
            single_trial_trajectory[1],
            "-",
            lw=linewidth_single_trial,
            c=color_single_trial,
            alpha=alpha_single_trial,
        )
    # trial averaged trajectory
    average_trajectory = np.mean(trajectories, axis=0)
    ax2.plot(
        average_trajectory[0],
        average_trajectory[1],
        "-",
        lw=linewidth_trial_average,
        c=color_trial_average,
        label="Trial averaged trajectory",
    )
    ax2.legend()

    plt.tight_layout()
    plt.show()


def plot_trajectories_all(trajectories_all):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.set_title("Latent dynamics extracted by GPFA")
    ax1.set_xlabel("Dim 1")
    ax1.set_ylabel("Dim 2")
    ax1.set_aspect(1)
    for single_trial_trajectory in trajectories_all:
        ax1.plot(
            single_trial_trajectory[0],
            single_trial_trajectory[1],
            "-",
            lw=linewidth_single_trial,
            c=color_single_trial,
            alpha=alpha_single_trial,
        )
    average_trajectory = np.mean(trajectories_all, axis=0)
    ax1.plot(
        average_trajectory[0],
        average_trajectory[1],
        "-",
        lw=linewidth_trial_average,
        c=color_trial_average,
        label="Trial averaged trajectory",
    )
    ax1.legend()

    trial_to_plot = 0
    ax2.set_title(f"Trajectory for trial {trial_to_plot}")
    ax2.set_xlabel("Time [s]")
    times_trajectory = np.arange(
        len(trajectories_all[trial_to_plot][0])
    ) * bin_size.rescale("s")
    ax2.plot(
        times_trajectory,
        trajectories_all[0][0],
        c="C0",
        label="Dim 1, fitting with all trials",
    )
    ax2.plot(
        times_trajectory,
        trajectories[0][0],
        c="C0",
        alpha=0.2,
        label="Dim 1, fitting with a half of trials",
    )
    ax2.plot(
        times_trajectory,
        trajectories_all[0][1],
        c="C1",
        label="Dim 2, fitting with all trials",
    )
    ax2.plot(
        times_trajectory,
        trajectories[0][1],
        c="C1",
        alpha=0.2,
        label="Dim 2, fitting with a half of trials",
    )
    ax2.legend()

    plt.tight_layout()
    plt.show()


def generate_synthetic_data():
    # set parameters for the integration of the harmonic oscillator
    timestep = 1 * pq.ms
    trial_duration = 2 * pq.s
    num_steps = int((trial_duration.rescale("ms") / timestep).magnitude)

    # set parameters for spike train generation
    max_rate = 70 * pq.Hz
    np.random.seed(
        42
    )  # for visualization purposes, we want to get identical spike trains at any run

    # specify data size
    num_trials = 20
    num_spiketrains = 50

    # generate a low-dimensional trajectory
    times_oscillator, oscillator_trajectory_2dim = integrated_oscillator(
        timestep.magnitude, num_steps=num_steps, x0=0, y0=1
    )
    # (2000,), (2, 2000)
    times_oscillator = (times_oscillator * timestep.units).rescale("s")
    # plt.plot(oscillator_trajectory_2dim[0][:750], oscillator_trajectory_2dim[1][:750])
    # random projection to high-dimensional space
    oscillator_trajectory_Ndim = random_projection(
        oscillator_trajectory_2dim, embedding_dimension=num_spiketrains
    )  # (50, 2000)
    # phase and amplitude are fluctuating

    # convert to instantaneous rate for Poisson process
    normed_traj = (
        oscillator_trajectory_Ndim / oscillator_trajectory_Ndim.max()
    )  # (50, 2000)
    instantaneous_rates_oscillator = np.power(max_rate.magnitude, normed_traj)

    # generate spike trains
    spiketrains_oscillator = generate_spiketrains(
        instantaneous_rates_oscillator, num_trials, timestep
    )
    # len(spiketrains_oscillator) # 20 # trials
    # len(spiketrains_oscillator[0]) # 50
    # len(spiketrains_oscillator[0][0]) # 4
    # print(spiketrains_oscillator[0])

    return (
        num_trials,
        num_spiketrains,
        times_oscillator,
        oscillator_trajectory_2dim,
        oscillator_trajectory_Ndim,
        spiketrains_oscillator,
    )


if __name__ == "__main__":
    # Generates synthetic data
    (
        num_trials,
        num_spiketrains,
        times_oscillator,
        oscillator_trajectory_2dim,
        oscillator_trajectory_Ndim,
        spiketrains_oscillator,
    ) = generate_synthetic_data()

    # plots
    linewidth_single_trial = 0.5
    plot_oscillator_trajectory_2dim(
        times_oscillator,
        oscillator_trajectory_2dim,
        num_spiketrains,
        spiketrains_oscillator,
    )

    # GPFA
    # specify fitting parameters
    bin_size = 20 * pq.ms
    latent_dimensionality = 2
    gpfa_2dim = GPFA(bin_size=bin_size, x_dim=latent_dimensionality)

    # spiketrains_oscillator.shape # (50, 2000)
    gpfa_2dim.fit(spiketrains_oscillator[: num_trials // 2])
    print(gpfa_2dim.params_estimated.keys())

    trajectories = gpfa_2dim.transform(spiketrains_oscillator[num_trials // 2 :])
    # trajectories[0].shape # (2, 100)
    plot_trajectories(trajectories)

    # only the half of trials
    # here we just reuse the existing instance of the GPFA() class as we use the same fitting parameters as before
    trajectories_all = gpfa_2dim.fit_transform(spiketrains_oscillator)
    plot_trajectories_all(trajectories_all)
