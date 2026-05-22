import argparse
from typing import Optional, Tuple

import numpy as np

from helper.trajectory_reader import read_trajectory
from helper.plot_functions import PlotFunctions as pf
from helper.utilities import low_pass, detect_anomalies_with_confidence, Normalize
from helper.detection_algo import *
from helper.cable_localisation import CableLocaliser, localise_cable

pf.PAPER_MODE = False

# ---------------------------------------------------------------------------- #
#                             script parameters                                 #
# ---------------------------------------------------------------------------- #

# Sampling
FS = 50.0
DT = 1.0 / FS
cutoff = 0.5

# Detection
DETECTION_METHOD = "CMF"
USE_NORM = True
TRIM_AMOUNT = 10

# Thresholds
THRESHOLD_PEAKS = 0.5
THRESHOLD_HOLES = 1.1

# Input
FILENAME = "trajectories/manip_simu/manipS45.csv"
SCALING_FACTOR = 0

# add false detections for testing
FAILING_PERSENTAGE = 50.0  # percentage of detection samples to randomly flip detection state of recorded samples (0-100)

# Localisation (Bharti et al. 2020, Section IV)
SIGMA_V   = 1.414   # observation noise std (m) — paper default
PSI0_DEG  = 0.0     # initial pipe heading guess (degrees)

# ---------------------------------------------------------------------------- #
#                              helper functions                                 #
# ---------------------------------------------------------------------------- #

def trim_and_merge(rx_filtered: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Suppress convolution edge artefacts and collapse filter outputs to a
    single normalised confidence trace.

    Args:
        rx_filtered: Raw filter output of shape (n_basis, n_channels, N).

    Returns:
        rx_filtered: Edge-corrected array.
        rx_merged:   (N,) normalised anomaly confidence trace.
    """
    if USE_NORM:
        rx_filtered = np.expand_dims(rx_filtered, axis=1)

    if TRIM_AMOUNT > 0:
        i = TRIM_AMOUNT
        rx_filtered[:, :, :i]  = rx_filtered[:, :, i : i + 1]
        rx_filtered[:, :, -i:] = rx_filtered[:, :, -i - 1 : -i]

    rx_merged = Normalize(np.linalg.norm(rx_filtered, axis=(0, 1)))
    return rx_filtered, rx_merged


def detect_cable(traj) -> np.ndarray:
    """
    Run the CMF anomaly detector and return indices of detected cable crossings.

    Returns
    -------
    anomaly_indexes : np.ndarray of int
        Trajectory sample indices flagged as cable crossings.
    """
    input_signal = low_pass(traj.mag_norm, cutoff, FS)
    input_signal = input_signal - np.mean(input_signal)

    drone_velocity = estimate_average_velocity(traj.longitude, traj.latitude, DT)
    print(f"Estimated average velocity: {drone_velocity:.2f} m/s")

    _, convolution_output = detect_anomalies(
        input_signal, traj.timestamp, method=DETECTION_METHOD,
        drone_velocity=drone_velocity,
    )
    rx_filtered, rx_merged = trim_and_merge(convolution_output)
    _, peak_anoms, hole_anoms = detect_anomalies_with_confidence(
        rx_merged,
        threshold_peaks=THRESHOLD_PEAKS,
        threshold_holes=THRESHOLD_HOLES,
    )
    anomaly_indexes = np.concatenate([peak_anoms, hole_anoms])
    print(f"Anomalies detected — peaks: {len(peak_anoms)}, holes: {len(hole_anoms)}")
    return anomaly_indexes


def build_centre_mask(n_samples: int, anomaly_indexes: np.ndarray) -> np.ndarray:
    """
    Convert a list of anomaly indices into a boolean detection mask.

    The detection algorithm flags crossings (the vehicle was above the cable at
    those timesteps), which corresponds to the 'Centre' detection state used by
    the EKF observation model.

    Parameters
    ----------
    n_samples       : int             — total number of trajectory samples
    anomaly_indexes : np.ndarray      — indices returned by detect_cable()

    Returns
    -------
    mask : np.ndarray of bool, shape (n_samples,)
    """
    mask = np.zeros(n_samples, dtype=bool)
    valid = anomaly_indexes[(anomaly_indexes >= 0) & (anomaly_indexes < n_samples)]
    mask[valid.astype(int)] = True
    return mask


# ---------------------------------------------------------------------------- #
#                                    main                                       #
# ---------------------------------------------------------------------------- #

def main():
    # ------------------------------------------------------------------ #
    # 1. Load trajectory
    # ------------------------------------------------------------------ #
    traj = read_trajectory(FILENAME, DT)
    traj.mag_norm = np.linalg.norm(traj.magnetic, axis=0)
    # ------------------------------------------------------------------ #
    # 2. Cable detection  (existing pipeline)
    # ------------------------------------------------------------------ #
    cable_indices = detect_cable(traj)
    
        # Optionally add false detections for testing
    if FAILING_PERSENTAGE > 0:
        n_samples = len(traj.timestamp)
        n_detections = len(cable_indices)
        n_to_flip = int(n_detections * FAILING_PERSENTAGE / 100)
        if n_to_flip > 0:
            flip_indices = np.random.choice(np.arange(n_samples), size=n_to_flip, replace=False)
            cable_indices = np.setxor1d(cable_indices, flip_indices)  # flip detection state
            print(f"Flipped {n_to_flip} detections for testing (total now: {len(cable_indices)})")
    # ------------------------------------------------------------------ #
    # 3. Cable localisation  (Bharti et al. 2020 EKF)
    # ------------------------------------------------------------------ #
    n = len(traj.timestamp)
    centre_mask = build_centre_mask(n, cable_indices)

    if centre_mask.sum() == 0:
        print("No centre detections — cannot initialise the localisation filter.")
        return

    print(f"\nRunning EKF localiser over {n} samples "
          f"({centre_mask.sum()} centre detections) …")

    localiser, x_cart, y_cart = localise_cable(
        longitude      = traj.longitude,
        latitude       = traj.latitude,
        detection_mask = centre_mask,
        sigma_v        = SIGMA_V,
        psi0           = np.deg2rad(PSI0_DEG),
    )

    pipe_x, pipe_y, heading_deg, _ = localiser.get_pipe_estimate()
    print(f"\nFinal pipe estimate:")
    print(f"  Position  : ({pipe_x:.2f} m E, {pipe_y:.2f} m N)  [Cartesian]")
    print(f"  Heading   : {heading_deg:.1f}°")
    print(f"  Updates   : {len(localiser.updated_steps)} EKF corrections applied")

    # Collect per-step estimates for plotting
    states = np.array(localiser.state_history)   # (M, 3)  M = steps run + 1

    # ------------------------------------------------------------------ #
    # 4. Plot
    # ------------------------------------------------------------------ #
    pf.plot_trajectory(
        traj.longitude, traj.latitude, traj.mag_norm,
        cable_indices,
    )

    # Plot pipe estimate track over the vehicle trajectory
    _plot_localisation(
        x_cart, y_cart, states, localiser, centre_mask,
    )


def _plot_localisation(
    x_cart:      np.ndarray,
    y_cart:      np.ndarray,
    states:      np.ndarray,
    localiser:   CableLocaliser,
    centre_mask: np.ndarray,
) -> None:
    """
    Plot vehicle trajectory, EKF pipe-position estimates, heading, and
    2-σ covariance ellipses at the final estimate.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import Ellipse

    covs = np.array(localiser.cov_history)   # (M, 3, 3)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- left: 2-D map ---------------------------------------------------
    ax = axes[0]
    ax.plot(x_cart, y_cart, "k.", ms=2, label="Vehicle")
    ax.plot(x_cart[centre_mask], y_cart[centre_mask],
            "ro", ms=4, label="Centre detections")

    n_states = len(states)
    idx = np.arange(len(states))

    sc = ax.scatter(
        states[:, 0],
        states[:, 1],
        c=idx,
        cmap="viridis",   # choose any matplotlib colormap
        s=10,
        label="Pipe estimate"
    )

    plt.colorbar(sc, ax=ax, label="Point index")
    ax.plot(states[-1, 0], states[-1, 1], "b*", ms=12)

    # 2-σ ellipse at final estimate
    P_xy = covs[-1][:2, :2]
    vals, vecs = np.linalg.eigh(P_xy)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * 2 * np.sqrt(vals)   # 2-σ
    ell = Ellipse(
        xy=(states[-1, 0], states[-1, 1]),
        width=width, height=height, angle=angle,
        edgecolor="blue", facecolor="none", linestyle="--", lw=1, label="2σ ellipse",
    )
    ax.add_patch(ell)

    # heading arrow
    psi = states[-1, 2]
    arrow_len = max(np.hypot(np.ptp(x_cart), np.ptp(y_cart)) * 0.1, 2.0)
    ax.annotate(
        "", xy=(states[-1, 0] + arrow_len * np.cos(psi),
                states[-1, 1] + arrow_len * np.sin(psi)),
        xytext=(states[-1, 0], states[-1, 1]),
        arrowprops=dict(arrowstyle="->", color="blue", lw=2),
    )

    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("Cable localisation — 2-D map")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")

    # --- right: heading estimate over time --------------------------------
    ax2 = axes[1]
    headings_deg = np.rad2deg(states[:, 2]) % 360
    std_psi_deg  = np.rad2deg(np.sqrt(covs[:, 2, 2]))

    t = np.arange(len(states))
    ax2.plot(t, headings_deg, "b-", label="ψ estimate")
    ax2.fill_between(
        t,
        headings_deg - 2 * std_psi_deg,
        headings_deg + 2 * std_psi_deg,
        alpha=0.2, color="blue", label="ψ ± 2σ",
    )
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Bearing ψ (degrees)")
    ax2.set_title("Pipe bearing estimate")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show(block=False)


if __name__ == "__main__":
    main()
    input("Press Enter to exit...")