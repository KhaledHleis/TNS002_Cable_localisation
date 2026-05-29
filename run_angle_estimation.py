import argparse
from typing import Optional, Tuple

import numpy as np

from helper.trajectory_reader_advanced import read_trajectory_advanced
from helper.plot_functions import PlotFunctions as pf
from helper.utilities import low_pass, detect_anomalies_with_confidence, Normalize
from helper.detection_algo import *
from helper.angle_estimator import *

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
THETA_PRIOR = 45

# Thresholds
THRESHOLD_PEAKS = 0.5
THRESHOLD_HOLES = 1.1
CORRESPONDING_WINDOW_SIZE = FS * 4  # FS by the time in seconds
# Input
FILENAME = "trajectories/manip_simu/manipS45_dual.csv"
SCALING_FACTOR = 0


# region #! helper functions
# ---------------------------------------------------------------------------- #
# ----------------------------- helper functions ----------------------------- #
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
        rx_filtered[:, :, :i] = rx_filtered[:, :, i : i + 1]
        rx_filtered[:, :, -i:] = rx_filtered[:, :, -i - 1 : -i]

    rx_merged = Normalize(np.linalg.norm(rx_filtered, axis=(0, 1)))
    return rx_filtered, rx_merged


def detect_cable(traj, sensor: str | int = "default") -> np.ndarray:
    """
    Run the CMF anomaly detector and return indices of detected cable crossings.

    Parameters
    ----------
    traj : Trajectory
        Parsed trajectory object.
    sensor : str or int, optional
        Sensor name (e.g. ``"UNO"``) or positional index (e.g. ``0``).
        Defaults to ``"default"`` for legacy single-sensor files.

    Returns
    -------
    anomaly_indexes : np.ndarray of int
        Trajectory sample indices flagged as cable crossings.
    """
    mag_norm = traj.sensors[sensor].norm

    input_signal = low_pass(mag_norm, cutoff, FS)
    input_signal = input_signal - np.mean(input_signal)

    drone_velocity = estimate_average_velocity(traj.longitude, traj.latitude, DT)
    print(f"Estimated average velocity: {drone_velocity:.2f} m/s")

    _, convolution_output = detect_anomalies(
        input_signal,
        traj.timestamp,
        method=DETECTION_METHOD,
        drone_velocity=drone_velocity,
        theta=THETA_PRIOR
        
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


def find_corresponding_detections(indices_1, indices_2, window_size):
    """
    Returns all pairs (i, j) where abs(i - j) <= window_size.
    """
    pairs = [(i, j) for i in indices_1 for j in indices_2 if abs(i - j) <= window_size]
    return pairs


# endregion


def main():
    traj = read_trajectory_advanced(FILENAME, DT)

    assert traj.n_sensors == 2, "Expected exactly 2 sensors for this example."

    sensor_names = traj.sensor_names
    print(f"Loaded trajectory with sensors: {sensor_names}")

    # detection algo on each sensor
    cable_indices_1 = detect_cable(traj, sensor=sensor_names[0])
    cable_indices_2 = detect_cable(traj, sensor=sensor_names[1])
    # from detection compute the angle predictions
    drone_velocity = estimate_average_velocity(traj.longitude, traj.latitude,DT)
    pairs = find_corresponding_detections(
        cable_indices_1, cable_indices_2, CORRESPONDING_WINDOW_SIZE
    )

    # Unzip pairs into two equal-length aligned arrays
    if pairs:
        indices_1_paired = np.array([p[0] for p in pairs])
        indices_2_paired = np.array([p[1] for p in pairs])
        print("velocity near the second detection")
        i = indices_1_paired[1]
        amount = 30
        drone_velocity_local = estimate_average_velocity(traj.longitude[i-amount:i+amount], traj.latitude[i-amount:i+amount],DT)
        print(drone_velocity_local)
        print(f"distances: \n {indices_2_paired-indices_1_paired}")
    else:
        print("No corresponding detections found.")
        return

    angles = []
    for s1,s2 in zip(indices_1_paired,indices_2_paired):
        amount = 30
        local_velocity = estimate_average_velocity(traj.longitude[s1-amount:s1+amount], traj.latitude[s1-amount:s1+amount],DT)
        angles.append(estimate_angle(s1,s2,Fs=FS,drone_velocity=local_velocity))
    angles = 90 - np.array(angles)
    # batch_estimate_angles(
    #     indices_1_paired, indices_2_paired, Fs=FS, drone_velocity=drone_velocity_local
    # )
    print(f"angles estimated from this trajectory: \n {angles}")


if __name__ == "__main__":
    main()
    input("Press Enter to exit...")
