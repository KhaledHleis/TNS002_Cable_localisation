"""
heading_estimator.py
====================
Estimates the drone's 2-D horizontal heading from raw magnetometer data,
and provides a utility to overlay heading vectors on a trajectory plot.

Convention
----------
    heading φ  — angle from East, counter-clockwise (radians)
    φ = atan2(By, Bx)  in the horizontal plane after low-pass filtering.

Notes
-----
- The raw mag signal contains cable anomalies; we low-pass filter first at
  the same cutoff used for detection so that dipole peaks do not bias the
  heading.
- The result is the *drone* heading (where the nose points), not a bearing
  to the cable.  The cable bearing is recovered as  ψ = φ ± α.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


# ---------------------------------------------------------------------------
# Low-pass helper (mirrors helper.utilities.low_pass to avoid circular import)
# ---------------------------------------------------------------------------

def _lowpass(signal: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter."""
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, signal)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_heading_from_mag(
    magnetic: np.ndarray,
    fs: float = 50.0,
    cutoff: float = 0.5,
) -> np.ndarray:
    """
    Estimate drone heading (radians) from a 3-axis magnetometer array.

    Parameters
    ----------
    magnetic : np.ndarray, shape (3, N)
        Raw magnetometer readings [Bx, By, Bz] in the drone body frame.
        Assumes Bx points forward (north-ish) and By points right (east-ish)
        in the horizontal plane.
    fs      : float — sampling frequency (Hz)
    cutoff  : float — low-pass cutoff (Hz); same as detection pipeline

    Returns
    -------
    heading : np.ndarray, shape (N,)
        Drone heading in radians, East=0, CCW positive.
        Wrapped to (-π, π].
    """
    Bx = _lowpass(magnetic[0], cutoff, fs)
    By = _lowpass(magnetic[1], cutoff, fs)

    heading = np.arctan2(By, Bx)          # (-π, π]
    return heading


def smooth_heading(heading: np.ndarray, window: int = 51) -> np.ndarray:
    """
    Additional median smoothing to remove residual spikes near cable anomalies.

    Parameters
    ----------
    heading : np.ndarray, shape (N,)
    window  : int — median filter half-window (samples); must be odd

    Returns
    -------
    smoothed : np.ndarray, shape (N,)
    """
    from scipy.signal import medfilt
    if window % 2 == 0:
        window += 1
    return medfilt(heading, kernel_size=window)


def resolve_heading_ambiguity(
    psi_candidate: float,
    psi_current:   float,
    alpha_rad:     float | None = None,
) -> float:
    """
    The angle observation  ψ_obs = φ - α  has a 180° ambiguity: both
    ψ_obs and ψ_obs + π are geometrically valid cable headings.

    Resolution: pick whichever of {ψ_obs, ψ_obs + π} is closest to the
    current EKF heading estimate ψ_current.  This is reliable as long as
    ψ_current is initialised close to the true cable heading (within 90°),
    which is guaranteed by setting PSI0_DEG ≈ known cable heading.

    The sign-based strategy (using sign of α to select branch) was removed
    because after normalising α to always-positive via sensor order swap,
    the sign no longer carries geometric meaning.

    Parameters
    ----------
    psi_candidate : float        — raw ψ_obs = φ_k − α_k  (radians)
    psi_current   : float        — current EKF heading estimate (radians)
    alpha_rad     : float | None — kept for API compatibility, unused

    Returns
    -------
    psi_resolved : float — disambiguated cable heading (radians), wrapped to (−π, π]
    """
    c0 = _wrap(psi_candidate)
    c1 = _wrap(psi_candidate + np.pi)

    # Always use proximity to current estimate — robust once PSI0 is set correctly
    if abs(_wrap(c0 - psi_current)) <= abs(_wrap(c1 - psi_current)):
        return c0
    return c1


def _wrap(angle: float) -> float:
    """Wrap angle to (-π, π]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# Visualisation helper
# ---------------------------------------------------------------------------

def plot_heading_on_trajectory(
    x_cart: np.ndarray,
    y_cart: np.ndarray,
    heading: np.ndarray,
    ax,
    n_arrows: int = 30,
    arrow_scale: float | None = None,
    color: str = "darkorange",
    label: str = "Drone heading",
) -> None:
    """
    Overlay evenly-spaced heading arrows on an existing trajectory axis.

    Parameters
    ----------
    x_cart, y_cart : np.ndarray, shape (N,)  — Cartesian vehicle positions
    heading        : np.ndarray, shape (N,)  — drone heading (radians)
    ax             : matplotlib Axes
    n_arrows       : int   — number of arrows to draw
    arrow_scale    : float — arrow length in metres; auto if None
    color          : str
    label          : str
    """
    if arrow_scale is None:
        extent     = max(np.ptp(x_cart), np.ptp(y_cart))
        arrow_scale = max(extent / (n_arrows * 1.5), 0.5)

    step    = max(len(x_cart) // n_arrows, 1)
    indices = np.arange(0, len(x_cart), step)

    for idx, i in enumerate(indices):
        dx = arrow_scale * np.cos(heading[i])
        dy = arrow_scale * np.sin(heading[i])
        ax.annotate(
            "",
            xy     = (x_cart[i] + dx, y_cart[i] + dy),
            xytext = (x_cart[i],       y_cart[i]),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
            label  = label if idx == 0 else None,
        )

    # Invisible scatter for legend entry
    ax.scatter([], [], marker=">", color=color, label=label, s=30)


def estimate_heading_from_gps(
    longitude:     np.ndarray,
    latitude:      np.ndarray,
    dt:            float,
    smooth_window: int = 21,
) -> np.ndarray:
    """
    Estimate drone heading (radians) from GPS position differences.

    This is the recommended heading source — immune to magnetic interference
    from drone motors and cable anomalies.

    Convention: East=0, CCW positive (same as ψ in the EKF).

    Parameters
    ----------
    longitude, latitude : (N,) decimal degrees
    dt                  : float — sample period (s)
    smooth_window       : int   — median filter window (samples, odd)

    Returns
    -------
    heading : (N,) radians, East=0, CCW
    """
    from scipy.signal import medfilt
    from helper.utilities import latlon_to_cartesian

    x, y = latlon_to_cartesian(longitude, latitude)

    # Central differences (np.gradient handles edges with one-sided differences)
    dx = np.gradient(x)
    dy = np.gradient(y)

    heading = np.arctan2(dy, dx)   # East=0, CCW — matches EKF ψ convention

    # Median filter to suppress GPS noise spikes
    if smooth_window % 2 == 0:
        smooth_window += 1
    heading = medfilt(heading, kernel_size=smooth_window)

    return heading