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

# Add false detections for testing
FAILING_PERSENTAGE = 0.0  # percentage of detection samples to randomly flip (0-100)

# Localisation (Bharti et al. 2020, Section IV)
SIGMA_V   = 1.414   # observation noise std (m)
PSI0_DEG  = 0.0     # initial cable heading guess (degrees)

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

    Parameters
    ----------
    n_samples       : int        — total number of trajectory samples
    anomaly_indexes : np.ndarray — indices returned by detect_cable()

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
        n_samples    = len(traj.timestamp)
        n_detections = len(cable_indices)
        n_to_flip    = int(n_detections * FAILING_PERSENTAGE / 100)
        if n_to_flip > 0:
            flip_indices  = np.random.choice(np.arange(n_samples), size=n_to_flip, replace=False)
            cable_indices = np.setxor1d(cable_indices, flip_indices)
            print(f"Flipped {n_to_flip} detections for testing "
                  f"(total now: {len(cable_indices)})")

    # ------------------------------------------------------------------ #
    # 3. Cable localisation  (Bharti et al. 2020 EKF — cable frame)
    # ------------------------------------------------------------------ #
    n           = len(traj.timestamp)
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
    print(f"\nFinal cable estimate (world Cartesian):")
    print(f"  Position  : ({pipe_x:.2f} m E, {pipe_y:.2f} m N)")
    print(f"  Heading   : {heading_deg:.1f}°")
    print(f"  Updates   : {len(localiser.updated_steps)} EKF corrections applied")

    # state_history[:, :2] = vehicle offset from cable (cable frame)
    # Recover cable world position at each step:
    #   cable_world_k = vehicle_world_k - state_k[:2]
    states = np.array(localiser.state_history)   # (M, 3)

    # Align vehicle trajectory with state history length.
    # state_history[0] is the init state (at first_det); subsequent entries
    # correspond to steps first_det+1 … N-1.
    first_det = int(np.argmax(centre_mask))
    n_states  = len(states)
    veh_x_aligned = x_cart[first_det : first_det + n_states]
    veh_y_aligned = y_cart[first_det : first_det + n_states]

    # Cable world positions at every step
    cable_x_hist = veh_x_aligned - states[:, 0]
    cable_y_hist = veh_y_aligned - states[:, 1]

    # ------------------------------------------------------------------ #
    # 4. Plot
    # ------------------------------------------------------------------ #
    pf.plot_trajectory(
        traj.longitude, traj.latitude, traj.mag_norm,
        cable_indices,
    )

    _plot_localisation(
        x_cart, y_cart,
        states, cable_x_hist, cable_y_hist,
        localiser, centre_mask,
    )


def _plot_localisation(
    x_cart:       np.ndarray,
    y_cart:       np.ndarray,
    states:       np.ndarray,   # (M, 3)  cable-frame: [dx, dy, ψ]
    cable_x_hist: np.ndarray,   # (M,)    cable world-x at each step
    cable_y_hist: np.ndarray,   # (M,)    cable world-y at each step
    localiser:    CableLocaliser,
    centre_mask:  np.ndarray,
) -> None:
    """
    Four-panel figure:
      Top-left  : vehicle trajectory + centre detections + cable world-position track
      Top-right : cable-frame offsets (dx, dy) over time — should converge to 0
      Bottom-left: heading estimate ψ with 2σ band
      Bottom-right: 2-D zoom on final cable position with covariance ellipse
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    covs = np.array(localiser.cov_history)   # (M, 3, 3)
    t    = np.arange(len(states))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # ── top-left: world map ──────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(x_cart, y_cart, "k.", ms=2, label="Vehicle trajectory")
    ax.plot(x_cart[centre_mask], y_cart[centre_mask],
            "ro", ms=5, label="Centre detections")

    sc = ax.scatter(
        cable_x_hist, cable_y_hist,
        c=t, cmap="viridis", s=8, label="Cable position (world)",
    )
    plt.colorbar(sc, ax=ax, label="Step index")
    ax.plot(cable_x_hist[-1], cable_y_hist[-1], "b*", ms=12, label="Final cable pos")

    # 2-σ ellipse at final cable position (uses P_xy from covariance)
    P_xy = covs[-1][:2, :2]
    vals, vecs = np.linalg.eigh(P_xy)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle  = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width  = 2 * 2 * np.sqrt(vals[0])
    height = 2 * 2 * np.sqrt(vals[1])
    ell = Ellipse(
        xy=(cable_x_hist[-1], cable_y_hist[-1]),
        width=width, height=height, angle=angle,
        edgecolor="blue", facecolor="none", linestyle="--", lw=1.5,
        label="2σ ellipse",
    )
    ax.add_patch(ell)

    # Heading arrow at final cable position
    psi       = states[-1, 2]
    arrow_len = max(np.hypot(np.ptp(x_cart), np.ptp(y_cart)) * 0.08, 1.0)
    ax.annotate(
        "", xy=(cable_x_hist[-1] + arrow_len * np.cos(psi),
                cable_y_hist[-1] + arrow_len * np.sin(psi)),
        xytext=(cable_x_hist[-1], cable_y_hist[-1]),
        arrowprops=dict(arrowstyle="->", color="blue", lw=2),
    )

    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("Cable localisation — world map")
    ax.legend(fontsize=7)
    ax.set_aspect("equal")

    # ── top-right: cable-frame offsets (dx, dy) ─────────────────────────
    ax2 = axes[0, 1]
    std_dx = np.sqrt(covs[:, 0, 0])
    std_dy = np.sqrt(covs[:, 1, 1])

    ax2.plot(t, states[:, 0], "b-",  label="dx (E offset)")
    ax2.fill_between(t,
                     states[:, 0] - 2 * std_dx,
                     states[:, 0] + 2 * std_dx,
                     alpha=0.2, color="blue")

    ax2.plot(t, states[:, 1], "r-",  label="dy (N offset)")
    ax2.fill_between(t,
                     states[:, 1] - 2 * std_dy,
                     states[:, 1] + 2 * std_dy,
                     alpha=0.2, color="red")

    ax2.axhline(0, color="k", lw=0.8, linestyle="--")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Offset from cable (m)")
    ax2.set_title("Vehicle offset from cable (cable frame) — should converge to 0")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # ── bottom-left: heading estimate ────────────────────────────────────
    ax3 = axes[1, 0]
    headings_deg = np.rad2deg(states[:, 2]) % 360
    std_psi_deg  = np.rad2deg(np.sqrt(covs[:, 2, 2]))

    ax3.plot(t, headings_deg, "b-", label="ψ estimate")
    ax3.fill_between(
        t,
        headings_deg - 2 * std_psi_deg,
        headings_deg + 2 * std_psi_deg,
        alpha=0.2, color="blue", label="ψ ± 2σ",
    )
    ax3.set_xlabel("Timestep")
    ax3.set_ylabel("Cable bearing ψ (degrees)")
    ax3.set_title("Cable heading estimate")
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)

    # ── bottom-right: zoom on final cable estimate ───────────────────────
    ax4 = axes[1, 1]
    zoom_margin = max(width, height, 1.0) * 3

    ax4.plot(x_cart, y_cart, "k.", ms=2, alpha=0.4, label="Vehicle")
    ax4.plot(x_cart[centre_mask], y_cart[centre_mask],
             "ro", ms=5, label="Centre detections")
    ax4.plot(cable_x_hist, cable_y_hist, "g-", lw=1, alpha=0.6, label="Cable track")
    ax4.plot(cable_x_hist[-1], cable_y_hist[-1], "b*", ms=14, label="Final estimate")

    ell2 = Ellipse(
        xy=(cable_x_hist[-1], cable_y_hist[-1]),
        width=width, height=height, angle=angle,
        edgecolor="blue", facecolor="lightblue", alpha=0.3,
        linestyle="--", lw=1.5, label="2σ ellipse",
    )
    ax4.add_patch(ell2)
    ax4.annotate(
        "", xy=(cable_x_hist[-1] + arrow_len * np.cos(psi),
                cable_y_hist[-1] + arrow_len * np.sin(psi)),
        xytext=(cable_x_hist[-1], cable_y_hist[-1]),
        arrowprops=dict(arrowstyle="->", color="blue", lw=2),
    )
    ax4.set_xlim(cable_x_hist[-1] - zoom_margin, cable_x_hist[-1] + zoom_margin)
    ax4.set_ylim(cable_y_hist[-1] - zoom_margin, cable_y_hist[-1] + zoom_margin)
    ax4.set_xlabel("East (m)")
    ax4.set_ylabel("North (m)")
    ax4.set_title("Final cable estimate — zoomed")
    ax4.legend(fontsize=7)
    ax4.set_aspect("equal")
    ax4.grid(True, alpha=0.3)

    plt.suptitle("EKF Cable Localisation (Cable-Frame)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show(block=False)


if __name__ == "__main__":
    main()
    input("Press Enter to exit...")