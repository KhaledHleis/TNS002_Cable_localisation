"""
main_dual.py
============
Full pipeline for dual-sensor cable localisation:

  1. Load dual-sensor trajectory
  2. Detect cable crossings on each sensor independently
  3. Estimate drone heading from the magnetic signal
  4. Match paired detections → estimate incidence angle α per crossing
  5. Build per-step alpha array for the EKF
  6. Run cable-frame EKF with  z_k = [0, 0, ψ_obs]  (or subsets)
  7. Plot: trajectory + heading vectors, offsets, heading estimate, final zoom
"""

from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from helper.trajectory_reader_advanced import read_trajectory_advanced
from helper.plot_functions import PlotFunctions as pf
from helper.utilities import low_pass, detect_anomalies_with_confidence, Normalize
from helper.detection_algo import *
from helper.angle_estimator import estimate_angle, batch_estimate_angles
from helper.cable_localisation_with_angle import CableLocaliser, localise_cable
from helper.heading_estimator import (
    estimate_heading_from_gps,
    plot_heading_on_trajectory,
)

pf.PAPER_MODE = False

# ---------------------------------------------------------------------------- #
#                             script parameters                                 #
# ---------------------------------------------------------------------------- #

FS            = 50.0
DT            = 1.0 / FS
cutoff        = 0.5

DETECTION_METHOD        = "CMF"
USE_NORM                = True
TRIM_AMOUNT             = 10
THETA_PRIOR             = 45
THRESHOLD_PEAKS         = 0.5
THRESHOLD_HOLES         = 1.1
CORRESPONDING_WINDOW_SIZE = FS * 4       # samples

FILENAME       = "trajectories/manip_simu/manipS45_dual.csv"
SENSOR_SEP     = 1.0                     # metres between the two sensors (l)

# EKF noise parameters
SIGMA_V        = 0.1                   # position obs noise std (m)
SIGMA_ALPHA    = np.deg2rad(15.0)        # angle obs noise std (rad) ≈15°
PSI0_DEG       = 45.0                    # initial heading guess — closer to 45° cable

# Heading estimation
HEADING_CUTOFF   = 0.5                   # Hz — low-pass before atan2
HEADING_MED_WIN  = 51                    # samples — median spike removal
N_HEADING_ARROWS = 30                    # arrows to draw on map


# ---------------------------------------------------------------------------- #
#                              helper functions                                 #
# ---------------------------------------------------------------------------- #

def trim_and_merge(rx_filtered: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if USE_NORM:
        rx_filtered = np.expand_dims(rx_filtered, axis=1)
    if TRIM_AMOUNT > 0:
        i = TRIM_AMOUNT
        rx_filtered[:, :, :i]  = rx_filtered[:, :, i : i + 1]
        rx_filtered[:, :, -i:] = rx_filtered[:, :, -i - 1 : -i]
    rx_merged = Normalize(np.linalg.norm(rx_filtered, axis=(0, 1)))
    return rx_filtered, rx_merged


def detect_cable(traj, sensor) -> np.ndarray:
    mag_norm     = traj.sensors[sensor].norm
    input_signal = low_pass(mag_norm, cutoff, FS)
    input_signal = input_signal - np.mean(input_signal)

    drone_velocity = estimate_average_velocity(traj.longitude, traj.latitude, DT)
    print(f"  [{sensor}] avg velocity: {drone_velocity:.2f} m/s")

    _, conv_out = detect_anomalies(
        input_signal, traj.timestamp,
        method=DETECTION_METHOD,
        drone_velocity=drone_velocity,
        theta=THETA_PRIOR,
    )
    _, rx_merged = trim_and_merge(conv_out)
    _, peak_anoms, hole_anoms = detect_anomalies_with_confidence(
        rx_merged,
        threshold_peaks=THRESHOLD_PEAKS,
        threshold_holes=THRESHOLD_HOLES,
    )
    idx = np.concatenate([peak_anoms, hole_anoms])
    print(f"  [{sensor}] detections — peaks: {len(peak_anoms)}, holes: {len(hole_anoms)}")
    return idx


def find_corresponding_detections(
    indices_1: np.ndarray,
    indices_2: np.ndarray,
    window_size: float,
) -> list[tuple[int, int]]:
    """Return all pairs (i, j) with |i - j| ≤ window_size."""
    return [
        (int(i), int(j))
        for i in indices_1
        for j in indices_2
        if abs(i - j) <= window_size
    ]


def build_centre_mask(n: int, indices: np.ndarray) -> np.ndarray:
    mask  = np.zeros(n, dtype=bool)
    valid = indices[(indices >= 0) & (indices < n)].astype(int)
    mask[valid] = True
    return mask


def build_alpha_array(
    n:                int,
    pairs:            list[tuple[int, int]],
    traj,
    amount:           int = 30,
) -> np.ndarray:
    """
    Build a per-sample alpha array (NaN where no estimate).

    For each matched pair we estimate α at the midpoint sample using a local
    velocity estimate around sensor-1's detection index.

    Returns
    -------
    alpha : np.ndarray (n,), dtype float64
        Angle α in radians; NaN at all non-detection samples.
    """
    alpha = np.full(n, np.nan, dtype=float)
    for s1, s2 in pairs:
        lo = max(s1 - amount, 0)
        hi = min(s1 + amount, n - 1)
        local_vel = estimate_average_velocity(
            traj.longitude[lo:hi], traj.latitude[lo:hi], DT
        )
        # Δt = (DUO_idx - UNO_idx) / Fs is negative because DUO leads physically.
        # Swap argument order so Δt = t_UNO - t_DUO > 0, giving α ∈ (0°, 90°].
        # If the sign ever flips (cable from other side), arctan2 handles it correctly.
        alpha_deg = estimate_angle(s2, s1, Fs=FS,
                                   drone_velocity=local_vel,
                                   l=SENSOR_SEP)
        midpoint  = (s1 + s2) // 2
        alpha[midpoint] = np.deg2rad(alpha_deg)
    return alpha


# ---------------------------------------------------------------------------- #
#                                    main                                       #
# ---------------------------------------------------------------------------- #

def main():
    # ------------------------------------------------------------------ #
    # 1. Load
    # ------------------------------------------------------------------ #
    traj = read_trajectory_advanced(FILENAME, DT)
    assert traj.n_sensors == 2, "Expected exactly 2 sensors."
    s0, s1 = traj.sensor_names
    print(f"Sensors: {s0}, {s1}")
    n = len(traj.timestamp)

    # ------------------------------------------------------------------ #
    # 2. Drone heading from magnetic signal
    #    Use sensor 0 (or average both — here sensor 0 for simplicity).
    # ------------------------------------------------------------------ #
    print("\nEstimating drone heading from GPS …")
    drone_heading = estimate_heading_from_gps(
        traj.longitude, traj.latitude, dt=DT, smooth_window=HEADING_MED_WIN,
    )
    print(f"  Heading range: [{np.rad2deg(drone_heading.min()):.1f}°, "
          f"{np.rad2deg(drone_heading.max()):.1f}°]")

    # ------------------------------------------------------------------ #
    # 3. Cable detection on each sensor
    # ------------------------------------------------------------------ #
    print("\nRunning cable detection …")
    idx_0 = detect_cable(traj, s0)
    idx_1 = detect_cable(traj, s1)

    # ------------------------------------------------------------------ #
    # 4. Pair detections + estimate incidence angles
    # ------------------------------------------------------------------ #
    pairs = find_corresponding_detections(idx_0, idx_1, CORRESPONDING_WINDOW_SIZE)
    print(f"\nPaired detections: {len(pairs)}")

    if not pairs:
        print("No paired detections — cannot estimate angles.")
        alpha_arr = None
    else:
        alpha_arr = build_alpha_array(n, pairs, traj)
        alpha_deg_valid = np.rad2deg(alpha_arr[~np.isnan(alpha_arr)])
        print(f"  α estimates (deg): {np.round(alpha_deg_valid, 1)}")
        print(f"  Expected range: (0°, 90°] for valid geometry")
        print(f"  ψ_obs = φ − α should be stable across passes (~{PSI0_DEG:.0f}°)")

    # ------------------------------------------------------------------ #
    # 5. Centre mask  (union of both sensors)
    # ------------------------------------------------------------------ #
    all_detections = idx_0.astype(int)   # S1 only — S2 offset means z=[0,0] is wrong for it
    centre_mask    = build_centre_mask(n, all_detections)
    print(f"\nCentre detections: {centre_mask.sum()}")

    if centre_mask.sum() == 0:
        print("No centre detections — aborting.")
        return

    # ------------------------------------------------------------------ #
    # 6. EKF localisation
    # ------------------------------------------------------------------ #
    print(f"\nRunning EKF over {n} samples …")
    localiser, x_cart, y_cart = localise_cable(
        longitude      = traj.longitude,
        latitude       = traj.latitude,
        detection_mask = centre_mask,
        drone_headings = drone_heading,
        alpha_per_step = alpha_arr,
        sigma_v        = SIGMA_V,
        sigma_alpha    = SIGMA_ALPHA,
        psi0           = np.deg2rad(PSI0_DEG),
    )

    pipe_x, pipe_y, hdg_deg, _ = localiser.get_pipe_estimate()
    print(f"\nFinal cable estimate:")
    print(f"  Position : ({pipe_x:.2f} m E, {pipe_y:.2f} m N)")
    print(f"  Heading  : {hdg_deg:.1f}°")
    print(f"  Position updates : {len(localiser.updated_steps)}")
    print(f"  Angle updates    : {len(localiser.angle_obs_steps)}")

    # ------------------------------------------------------------------ #
    # 7. Derive cable world-position history
    # ------------------------------------------------------------------ #
    states    = np.array(localiser.state_history)   # (M, 3)
    covs      = np.array(localiser.cov_history)     # (M, 3, 3)

    first_det      = int(np.argmax(centre_mask))
    n_states       = len(states)
    veh_x_aligned  = x_cart[first_det : first_det + n_states]
    veh_y_aligned  = y_cart[first_det : first_det + n_states]
    cable_x_hist   = veh_x_aligned - states[:, 0]
    cable_y_hist   = veh_y_aligned - states[:, 1]

    # ------------------------------------------------------------------ #
    # 8. Plot
    # ------------------------------------------------------------------ #
    _plot_heading_verification(
        x_cart, y_cart, drone_heading, centre_mask, idx_0, idx_1,
    )
    _plot_localisation(
        x_cart, y_cart,
        states, covs,
        cable_x_hist, cable_y_hist,
        localiser, centre_mask,
        drone_heading, alpha_arr,
    )


# ---------------------------------------------------------------------------- #
#                               plotting                                        #
# ---------------------------------------------------------------------------- #

def _plot_heading_verification(
    x_cart:       np.ndarray,
    y_cart:       np.ndarray,
    drone_heading: np.ndarray,
    centre_mask:  np.ndarray,
    idx_0:        np.ndarray,
    idx_1:        np.ndarray,
) -> None:
    """
    Verification plot: trajectory with heading arrows overlaid.
    Lets you sanity-check that atan2(By, Bx) matches the flight direction.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # left: map with heading arrows
    ax = axes[0]
    ax.plot(x_cart, y_cart, "k.", ms=2, alpha=0.5, label="Vehicle")
    ax.plot(x_cart[idx_0.astype(int)], y_cart[idx_0.astype(int)],
            "b^", ms=5, label=f"Sensor 0 detections")
    ax.plot(x_cart[idx_1.astype(int)], y_cart[idx_1.astype(int)],
            "rs", ms=5, label=f"Sensor 1 detections")

    plot_heading_on_trajectory(
        x_cart, y_cart, drone_heading, ax,
        n_arrows=N_HEADING_ARROWS,
        color="darkorange",
        label="Drone heading (mag)",
    )
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("Trajectory + drone heading (verify mag-derived heading)")
    ax.legend(fontsize=7)
    ax.set_aspect("equal")

    # right: heading signal over time
    ax2 = axes[1]
    t   = np.arange(len(drone_heading))
    ax2.plot(t, np.rad2deg(drone_heading), "darkorange", lw=1, label="φ (drone heading)")
    # mark detection samples
    det_idx = np.where(centre_mask)[0]
    ax2.scatter(det_idx, np.rad2deg(drone_heading[det_idx]),
                color="red", s=20, zorder=5, label="Centre detections")
    ax2.set_xlabel("Sample")
    ax2.set_ylabel("Heading φ (degrees)")
    ax2.set_title("Drone heading from magnetic signal")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Heading verification", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show(block=False)


def _plot_localisation(
    x_cart:        np.ndarray,
    y_cart:        np.ndarray,
    states:        np.ndarray,     # (M, 3)
    covs:          np.ndarray,     # (M, 3, 3)
    cable_x_hist:  np.ndarray,
    cable_y_hist:  np.ndarray,
    localiser:     CableLocaliser,
    centre_mask:   np.ndarray,
    drone_heading: np.ndarray,
    alpha_arr:     np.ndarray | None,
) -> None:
    """
    Four-panel localisation figure:
      [0,0] World map — cable track, detections, 2σ ellipse, heading arrow
      [0,1] Vehicle offset from cable (dx, dy) — should converge to 0
      [1,0] Heading estimate ψ with 2σ band; overlaid angle observations
      [1,1] Zoom on final cable estimate
    """
    t    = np.arange(len(states))
    psi  = states[-1, 2]

    # 2σ ellipse parameters (reused in both map panels)
    P_xy = covs[-1][:2, :2]
    vals, vecs = np.linalg.eigh(P_xy)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle_ell = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w_ell     = 2 * 2 * np.sqrt(vals[0])
    h_ell     = 2 * 2 * np.sqrt(vals[1])
    arrow_len = max(np.hypot(np.ptp(x_cart), np.ptp(y_cart)) * 0.08, 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # ── [0,0] world map ───────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(x_cart, y_cart, "k.", ms=2, alpha=0.4, label="Vehicle")
    ax.plot(x_cart[centre_mask], y_cart[centre_mask],
            "ro", ms=4, label="Centre detections")
    sc = ax.scatter(cable_x_hist, cable_y_hist, c=t, cmap="viridis",
                    s=8, label="Cable pos track")
    plt.colorbar(sc, ax=ax, label="Step")
    ax.plot(cable_x_hist[-1], cable_y_hist[-1], "b*", ms=12)
    ell = Ellipse(
        xy=(cable_x_hist[-1], cable_y_hist[-1]),
        width=w_ell, height=h_ell, angle=angle_ell,
        edgecolor="blue", facecolor="none", linestyle="--", lw=1.5,
        label="2σ ellipse",
    )
    ax.add_patch(ell)
    ax.annotate(
        "", xy=(cable_x_hist[-1] + arrow_len * np.cos(psi),
                cable_y_hist[-1] + arrow_len * np.sin(psi)),
        xytext=(cable_x_hist[-1], cable_y_hist[-1]),
        arrowprops=dict(arrowstyle="->", color="blue", lw=2),
    )
    ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
    ax.set_title("Cable localisation — world map")
    ax.legend(fontsize=7); ax.set_aspect("equal")

    # ── [0,1] cable-frame offsets ─────────────────────────────────────
    ax2 = axes[0, 1]
    std_dx = np.sqrt(covs[:, 0, 0])
    std_dy = np.sqrt(covs[:, 1, 1])
    ax2.plot(t, states[:, 0], "b-", label="dx (E offset)")
    ax2.fill_between(t, states[:, 0] - 2*std_dx, states[:, 0] + 2*std_dx,
                     alpha=0.2, color="blue")
    ax2.plot(t, states[:, 1], "r-", label="dy (N offset)")
    ax2.fill_between(t, states[:, 1] - 2*std_dy, states[:, 1] + 2*std_dy,
                     alpha=0.2, color="red")
    ax2.axhline(0, color="k", lw=0.8, linestyle="--")
    # mark angle-obs steps
    if localiser.angle_obs_steps:
        ao = np.array(localiser.angle_obs_steps)
        # map global step indices to state_history indices
        first_det = localiser.updated_steps[0] if localiser.updated_steps else 0
        ao_t = ao - first_det + 1
        ao_t = ao_t[(ao_t >= 0) & (ao_t < len(t))]
        ax2.vlines(ao_t, ax2.get_ylim()[0], ax2.get_ylim()[1],
                   color="green", alpha=0.3, lw=0.8, label="Angle obs")
    ax2.set_xlabel("Timestep"); ax2.set_ylabel("Offset from cable (m)")
    ax2.set_title("Vehicle offset from cable — should converge to 0")
    ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)

    # ── [1,0] heading estimate ────────────────────────────────────────
    ax3 = axes[1, 0]
    headings_deg = np.rad2deg(states[:, 2]) % 360
    std_psi_deg  = np.rad2deg(np.sqrt(covs[:, 2, 2]))
    ax3.plot(t, headings_deg, "b-", label="ψ estimate")
    ax3.fill_between(t,
                     headings_deg - 2 * std_psi_deg,
                     headings_deg + 2 * std_psi_deg,
                     alpha=0.2, color="blue", label="ψ ± 2σ")

    # overlay angle observations
    if alpha_arr is not None and localiser.angle_obs_steps:
        first_det = localiser.updated_steps[0] if localiser.updated_steps else 0
        for step in localiser.angle_obs_steps:
            t_idx = step - first_det + 1
            if 0 <= t_idx < len(t) and not np.isnan(alpha_arr[step]):
                ax3.scatter(t_idx,
                            np.rad2deg(alpha_arr[step]) % 360,
                            color="green", s=25, zorder=5)
        ax3.scatter([], [], color="green", s=25, label="α observation")

    ax3.set_xlabel("Timestep"); ax3.set_ylabel("Bearing ψ (degrees)")
    ax3.set_title("Cable heading estimate + angle observations")
    ax3.legend(fontsize=7); ax3.grid(True, alpha=0.3)

    # ── [1,1] zoom on final estimate ──────────────────────────────────
    ax4 = axes[1, 1]
    zoom = max(w_ell, h_ell, 1.0) * 3
    ax4.plot(x_cart, y_cart, "k.", ms=2, alpha=0.4, label="Vehicle")
    ax4.plot(x_cart[centre_mask], y_cart[centre_mask],
             "ro", ms=5, label="Centre detections")
    ax4.plot(cable_x_hist, cable_y_hist, "g-", lw=1, alpha=0.6,
             label="Cable track")
    ax4.plot(cable_x_hist[-1], cable_y_hist[-1], "b*", ms=14,
             label="Final estimate")
    ell2 = Ellipse(
        xy=(cable_x_hist[-1], cable_y_hist[-1]),
        width=w_ell, height=h_ell, angle=angle_ell,
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
    ax4.set_xlim(cable_x_hist[-1] - zoom, cable_x_hist[-1] + zoom)
    ax4.set_ylim(cable_y_hist[-1] - zoom, cable_y_hist[-1] + zoom)
    ax4.set_xlabel("East (m)"); ax4.set_ylabel("North (m)")
    ax4.set_title("Final cable estimate — zoomed")
    ax4.legend(fontsize=7); ax4.set_aspect("equal"); ax4.grid(True, alpha=0.3)

    plt.suptitle("EKF Cable Localisation — Dual Sensor + Angle Observation",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show(block=False)


if __name__ == "__main__":
    main()
    input("Press Enter to exit...")