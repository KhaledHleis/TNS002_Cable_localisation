"""
angle_estimator.py
==================
Estimates the angle between the drone's flight track and a cable from
the time difference between two cross-track magnetometer sensors.

Sensor layout
-------------
The two sensors (S1, S2) are mounted side-by-side, perpendicular to the
drone's flight direction, separated by distance l (metres).

                     flight direction →
                    ──────────────────────────
         S1 ●   (offset +l/2 from centre line)
                    ──────────────────────────
         S2 ●   (offset −l/2 from centre line)

Geometry
--------
When the drone flies at angle α relative to the cable, the cable crosses
the S1 and S2 sensor tracks at different along-track positions.  The
along-track separation of the two crossing points is Δt · v (metres), and
the cross-track separation is l (metres), giving:

    tan(α) = l / (Δt · v)
    α = atan2(l, Δt · v)

where α is the angle between the drone flight track and the cable (rad).

Sign convention
---------------
    Δt = (t_S2 − t_S1) = (index_S2 − index_S1) / Fs   [signed]

    Δt > 0  → S1 detects before S2  → cable is tilted so it crosses S1
               track first.  α > 0 (positive crossing angle).
    Δt < 0  → S2 detects first.  α < 0 (negative crossing angle).
    Δt = 0  → simultaneous (drone perpendicular to cable).  α = 90°.

Output convention
-----------------
    α ∈ (−90°, +90°]:  angle from cable direction to drone track.
    α = 90°   → perpendicular crossing (maximum sensitivity)
    α → 0°    → nearly parallel (approaching detachability limit)

    The sign of α is used inside the EKF to remove the 180° ambiguity
    in the cable heading estimate without needing a separate heuristic.

Notes
-----
- The previous version used abs(Δt), which always returned a positive
  angle in [0°, 90°] and discarded direction information.  This version
  preserves the sign.
- The previous version also applied an additional 90 − result transform
  inside the function, which the calling script then reversed with another
  90 − angles.  Both transforms have been removed; the function now returns
  α directly so there is no ambiguity about which convention is active.
"""

from __future__ import annotations

import numpy as np


def estimate_angle(
    detection_index_1: int,
    detection_index_2: int,
    Fs:                float,
    drone_velocity:    float = 1.0,
    l:                 float = 1.0,
) -> float:
    """
    Estimate the angle between the drone's flight track and the cable.

    Parameters
    ----------
    detection_index_1 : int   — sample index of S1 detection
    detection_index_2 : int   — sample index of S2 detection
    Fs                : float — sampling frequency (Hz)
    drone_velocity    : float — drone speed (m/s)
    l                 : float — cross-track sensor separation (m)

    Returns
    -------
    alpha : float
        Signed angle in degrees between drone track and cable.
        α ∈ (−90°, +90°].
        Positive: S1 detects before S2.
        Negative: S2 detects before S1.
        90°: simultaneous detection (perpendicular crossing).
    """
    # Signed time difference: positive if S1 detects first
    time_diff = (detection_index_2 - detection_index_1) / Fs   # seconds (signed)

    # atan2 handles time_diff = 0 correctly: returns 90°
    alpha = np.degrees(np.arctan2(l, time_diff * drone_velocity))
    return float(alpha)


def batch_estimate_angles(
    detection_indices_1: np.ndarray,
    detection_indices_2: np.ndarray,
    Fs:                  float,
    drone_velocity:      float = 1.0,
    l:                   float = 1.0,
) -> np.ndarray:
    """
    Batch version of estimate_angle.

    Parameters
    ----------
    detection_indices_1 : (N,) int array — S1 detection sample indices
    detection_indices_2 : (N,) int array — S2 detection sample indices
    Fs                  : float — sampling frequency (Hz)
    drone_velocity      : float — drone speed (m/s)
    l                   : float — cross-track sensor separation (m)

    Returns
    -------
    alphas : (N,) float array — signed angles in degrees
    """
    time_diffs = (detection_indices_2 - detection_indices_1) / Fs   # signed
    alphas     = np.degrees(np.arctan2(l, time_diffs * drone_velocity))
    return alphas