from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Trajectory:
    """
    Structured container for a parsed trajectory file.

    Attributes
    ----------
    timestamp : pd.Series, shape (N,)
        UTC timestamps.
    longitude : pd.Series, shape (N,)
        Longitude in decimal degrees.
    latitude : pd.Series, shape (N,)
        Latitude in decimal degrees.
    velocity : np.ndarray, shape (3, N)
        Body-frame velocity components [ve, vn, vd] in m/s.
    heading : pd.Series, shape (N,)
        Heading angle in degrees (0–360, clockwise from North).
    magnetic : np.ndarray, shape (3, N)
        Magnetic field vector [mag_x, mag_y, mag_z] in µT.
    mag_norm : pd.Series, shape (N,)
        Magnetic field norm in µT.
    acceleration : np.ndarray, shape (3, N)
        Acceleration vector [acc_x, acc_y, acc_z] in m/s².

    Notes
    -----
    Vector arrays follow the (3, N) convention so that a single component
    can be accessed as ``traj.velocity[0]`` (east), ``[1]`` (north),
    ``[2]`` (down), and the full vector at time step k as
    ``traj.velocity[:, k]``.
    """

    timestamp: pd.Series
    longitude: pd.Series
    latitude: pd.Series
    velocity: np.ndarray       # (3, N) — [ve, vn, vd]
    heading: pd.Series
    magnetic: np.ndarray       # (3, N) — [mag_x, mag_y, mag_z]
    mag_norm: pd.Series
    acceleration: np.ndarray   # (3, N) — [acc_x, acc_y, acc_z]

    def __len__(self) -> int:
        return len(self.timestamp)


# ---------------------------------------------------------------------------
# Required columns and their expected dtypes (used for validation).
# None means "keep the original dtype" (used for timestamp).
# ---------------------------------------------------------------------------
_F64 = np.dtype(np.float64)

_REQUIRED_COLUMNS: dict[str, Optional[np.dtype]] = {
    "timestamp": None,   # keep original dtype (datetime or int)
    "longitude": _F64,
    "latitude":  _F64,
    "ve":        _F64,
    "vn":        _F64,
    "vd":        _F64,
    "heading":   _F64,
    "mag_x":     _F64,
    "mag_y":     _F64,
    "mag_z":     _F64,
    "mag":       _F64,
    "acc_x":     _F64,
    "acc_y":     _F64,
    "acc_z":     _F64,
}


def _validate(df: pd.DataFrame, file_path: str) -> None:
    """Raise informative errors for missing columns or non-finite values."""
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Trajectory file '{file_path}' is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    numeric_cols = [c for c, dt in _REQUIRED_COLUMNS.items() if dt is not None]
    for col in numeric_cols:
        if not np.isfinite(df[col].to_numpy()).all():
            n_bad = (~np.isfinite(df[col].to_numpy())).sum()
            raise ValueError(
                f"Column '{col}' in '{file_path}' contains {n_bad} "
                f"non-finite value(s) (NaN or Inf)."
            )


def read_trajectory(file_path: str) -> Trajectory:
    """
    Read a trajectory CSV file and return a typed ``Trajectory`` object.

    Expected CSV columns (order-independent)::

        timestamp, longitude, latitude,
        ve, vn, vd,
        heading,
        mag_x, mag_y, mag_z, mag,
        acc_x, acc_y, acc_z

    Parameters
    ----------
    file_path : str
        Path to the trajectory CSV file.

    Returns
    -------
    Trajectory
        Dataclass holding each field as a named attribute.
        Vector quantities (velocity, magnetic, acceleration) are returned as
        ``np.ndarray`` of shape ``(3, N)``; scalar quantities are
        ``pd.Series`` of length ``N``.

    Raises
    ------
    FileNotFoundError
        If *file_path* does not exist.
    ValueError
        If required columns are absent or any numeric column contains
        non-finite values (NaN / Inf).

    Examples
    --------
    >>> traj = read_trajectory("data/run_01.csv")
    >>> print(len(traj), "samples")
    >>> east_velocity = traj.velocity[0]   # shape (N,)
    >>> full_vec_t0   = traj.velocity[:, 0]  # shape (3,)
    """
    df = pd.read_csv(file_path)

    _validate(df, file_path)

    # Cast numeric columns to float64 for downstream numerical consistency.
    numeric_cols = [c for c, dt in _REQUIRED_COLUMNS.items() if dt is not None]
    df[numeric_cols] = df[numeric_cols].astype(np.float64)

    return Trajectory(
        timestamp    = df["timestamp"],
        longitude    = df["longitude"],
        latitude     = df["latitude"],
        velocity     = np.stack((df["ve"], df["vn"], df["vd"]), axis=0),
        heading      = df["heading"],
        magnetic     = np.stack((df["mag_x"], df["mag_y"], df["mag_z"]), axis=0),
        mag_norm     = df["mag"],
        acceleration = np.stack((df["acc_x"], df["acc_y"], df["acc_z"]), axis=0),
    )