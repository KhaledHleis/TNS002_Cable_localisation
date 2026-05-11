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
    timestamp : np.ndarray, shape (N,)
        UTC timestamps.
    longitude : np.ndarray, shape (N,)
        Longitude in decimal degrees.
    latitude : np.ndarray, shape (N,)
        Latitude in decimal degrees.
    velocity : np.ndarray or None, shape (3, N)
        Body-frame velocity components [ve, vn, vd] in m/s, or ``None`` if
        the CSV does not contain velocity columns.
    heading : np.ndarray, shape (N,)
        Heading angle in degrees (0–360, clockwise from North).
    magnetic : np.ndarray, shape (3, N)
        Magnetic field vector [magx, magy, magz] in µT.
    mag_norm : np.ndarray, shape (N,)
        Magnetic field norm in µT.
    acceleration : np.ndarray or None, shape (3, N)
        Acceleration vector [acc_x, acc_y, acc_z] in m/s², or ``None`` if
        the CSV does not contain acceleration columns.

    Notes
    -----
    Vector arrays follow the (3, N) convention so that a single component
    can be accessed as ``traj.velocity[0]`` (east), ``[1]`` (north),
    ``[2]`` (down), and the full vector at time step k as
    ``traj.velocity[:, k]``.
    """

    timestamp: np.ndarray
    longitude: np.ndarray
    latitude: np.ndarray
    velocity: Optional[np.ndarray]      # (3, N) — [ve, vn, vd], or None
    heading: np.ndarray
    magnetic: np.ndarray                # (3, N) — [magx, magy, magz]
    mag_norm: np.ndarray
    acceleration: Optional[np.ndarray]  # (3, N) — [acc_x, acc_y, acc_z], or None

    def __len__(self) -> int:
        return len(self.timestamp)


# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------
_F64 = np.dtype(np.float64)

_REQUIRED_COLUMNS: dict[str, Optional[np.dtype]] = {
    "timestamp": None,   # keep original dtype (datetime or int)
    "longitude": _F64,
    "latitude":  _F64,
    "heading":   _F64,
    "magx":      _F64,
    "magy":      _F64,
    "magz":      _F64,
    "mag":       _F64,
}

_OPTIONAL_COLUMNS: dict[str, np.dtype] = {
    "ve":    _F64,
    "vn":    _F64,
    "vd":    _F64,
    "acc_x": _F64,
    "acc_y": _F64,
    "acc_z": _F64,
}

# Optional columns that must all be present together or all absent.
_OPTIONAL_GROUPS: dict[str, tuple[str, ...]] = {
    "velocity":     ("ve", "vn", "vd"),
    "acceleration": ("acc_x", "acc_y", "acc_z"),
}


def _validate(df: pd.DataFrame, file_path: str) -> None:
    """Raise informative errors for missing columns or non-finite values."""
    # --- required columns ---------------------------------------------------
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Trajectory file '{file_path}' is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    numeric_required = [c for c, dt in _REQUIRED_COLUMNS.items() if dt is not None]
    for col in numeric_required:
        if not np.isfinite(df[col].to_numpy()).all():
            n_bad = int((~np.isfinite(df[col].to_numpy())).sum())
            raise ValueError(
                f"Column '{col}' in '{file_path}' contains {n_bad} "
                f"non-finite value(s) (NaN or Inf)."
            )

    # --- optional columns present in this file ------------------------------
    for col in (c for c in _OPTIONAL_COLUMNS if c in df.columns):
        if not np.isfinite(df[col].to_numpy()).all():
            n_bad = int((~np.isfinite(df[col].to_numpy())).sum())
            raise ValueError(
                f"Column '{col}' in '{file_path}' contains {n_bad} "
                f"non-finite value(s) (NaN or Inf)."
            )

    # --- partial-group guard ------------------------------------------------
    for group_name, cols in _OPTIONAL_GROUPS.items():
        present = [c for c in cols if c in df.columns]
        if 0 < len(present) < len(cols):
            missing_cols = [c for c in cols if c not in df.columns]
            raise ValueError(
                f"Trajectory file '{file_path}' contains partial {group_name} "
                f"columns. Expected all of {list(cols)} or none. "
                f"Missing: {missing_cols}"
            )


def read_trajectory(file_path: str, DT: float) -> Trajectory:
    """
    Read a trajectory CSV file and return a typed ``Trajectory`` object.

    Required CSV columns (order-independent)::

        timestamp, longitude, latitude, heading,
        magx, magy, magz, mag

    Optional column groups (all or none)::

        ve, vn, vd          → velocity  (None when absent)
        acc_x, acc_y, acc_z → acceleration  (None when absent)

    Parameters
    ----------
    file_path : str
        Path to the trajectory CSV file.

    Returns
    -------
    Trajectory
        Dataclass holding each field as a named attribute.
        Vector quantities are ``np.ndarray`` of shape ``(3, N)`` when present,
        or ``None`` when the corresponding columns are absent from the file.
        Scalar quantities are ``pd.Series`` of length ``N``.

    Raises
    ------
    FileNotFoundError
        If *file_path* does not exist.
    ValueError
        If required columns are absent, an optional group is only partially
        present, or any numeric column contains non-finite values (NaN / Inf).

    Examples
    --------
    >>> traj = read_trajectory("data/run_01.csv", DT=1.0)
    >>> print(len(traj), "samples")
    >>> if traj.velocity is not None:
    ...     east_velocity = traj.velocity[0]   # shape (N,)
    """
    df = pd.read_csv(file_path)

    _validate(df, file_path)

    # Cast all present numeric columns to float64.
    cols_to_cast = [
        c for c, dt in {**_REQUIRED_COLUMNS, **_OPTIONAL_COLUMNS}.items()
        if dt is not None and c in df.columns
    ]
    df[cols_to_cast] = df[cols_to_cast].astype(np.float64)

    has_velocity     = all(c in df.columns for c in _OPTIONAL_GROUPS["velocity"])
    has_acceleration = all(c in df.columns for c in _OPTIONAL_GROUPS["acceleration"])

    return Trajectory(
        timestamp    = df["timestamp"].index.to_numpy() * DT,
        longitude    = df["longitude"].to_numpy(),
        latitude     = df["latitude"].to_numpy(),
        velocity     = (
            np.stack((df["ve"], df["vn"], df["vd"]), axis=0)
            if has_velocity else None
        ),
        heading      = df["heading"].to_numpy(),
        magnetic     = np.stack((df["magx"], df["magy"], df["magz"]), axis=0),
        mag_norm     = df["mag"].to_numpy(),
        acceleration = (
            np.stack((df["acc_x"], df["acc_y"], df["acc_z"]), axis=0)
            if has_acceleration else None
        ),
    )