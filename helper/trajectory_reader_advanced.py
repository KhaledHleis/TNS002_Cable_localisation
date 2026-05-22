from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import re

import numpy as np
import pandas as pd


@dataclass
class MagneticSensor:
    """
    Data for a single magnetometer sensor.

    Attributes
    ----------
    field : np.ndarray, shape (3, N)
        Magnetic field vector [magx, magy, magz] in µT.
    norm : np.ndarray, shape (N,)
        Magnetic field norm in µT.
    """
    field: np.ndarray   # (3, N)
    norm:  np.ndarray   # (N,)


@dataclass
class Trajectory:
    """
    Structured container for a parsed trajectory file supporting multiple
    magnetometer sensors.

    Attributes
    ----------
    timestamp : np.ndarray, shape (N,)
        UTC timestamps derived from row index * DT.
    longitude : np.ndarray, shape (N,)
        Longitude in decimal degrees.
    latitude : np.ndarray, shape (N,)
        Latitude in decimal degrees.
    heading : np.ndarray, shape (N,)
        Heading angle in degrees (0–360, clockwise from North).
    sensors : dict[str, MagneticSensor]
        Magnetic data keyed by sensor name (e.g. ``"UNO"``, ``"DUO"``).
        Single-sensor files use the key ``"default"``.
    velocity : np.ndarray or None, shape (3, N)
        Body-frame velocity [ve, vn, vd] in m/s, or ``None`` if absent.
    acceleration : np.ndarray or None, shape (3, N)
        Acceleration [acc_x, acc_y, acc_z] in m/s², or ``None`` if absent.

    Examples
    --------
    Access a specific sensor::

        b_uno = traj.sensors["UNO"].field       # shape (3, N)
        norm_duo = traj.sensors["DUO"].norm     # shape (N,)

    Iterate over all sensors::

        for name, sensor in traj.sensors.items():
            process(name, sensor.field)

    Stack all sensor norms into a (S, N) array::

        norms = np.stack([s.norm for s in traj.sensors.values()])
    """

    timestamp:    np.ndarray
    longitude:    np.ndarray
    latitude:     np.ndarray
    heading:      np.ndarray
    sensors:      dict[str, MagneticSensor]
    velocity:     Optional[np.ndarray] = field(default=None)  # (3, N)
    acceleration: Optional[np.ndarray] = field(default=None)  # (3, N)

    def __len__(self) -> int:
        return len(self.timestamp)

    @property
    def sensor_names(self) -> list[str]:
        """Ordered list of sensor names present in this trajectory."""
        return list(self.sensors.keys())

    @property
    def n_sensors(self) -> int:
        """Number of magnetometer sensors."""
        return len(self.sensors)

    @property
    def magnetic(self) -> np.ndarray:
        """
        All sensor fields stacked into a single array of shape (S, 3, N),
        where S = number of sensors, in ``sensor_names`` order.
        Convenience property for vectorised multi-sensor processing.
        """
        return np.stack([s.field for s in self.sensors.values()])

    @property
    def mag_norm(self) -> np.ndarray:
        """
        All sensor norms stacked into shape (S, N), in ``sensor_names`` order.
        """
        return np.stack([s.norm for s in self.sensors.values()])


# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------
_F64 = np.dtype(np.float64)

_BASE_REQUIRED: dict[str, np.dtype | None] = {
    "timestamp": None,
    "longitude": _F64,
    "latitude":  _F64,
    "heading":   _F64,
}

_OPTIONAL_GROUPS: dict[str, tuple[str, ...]] = {
    "velocity":     ("ve", "vn", "vd"),
    "acceleration": ("acc_x", "acc_y", "acc_z"),
}

# Patterns for legacy single-sensor column names (no suffix)
_LEGACY_MAG_COLS = ("magx", "magy", "magz", "mag")

# Pattern to detect multi-sensor magnetic columns: magx_sensor_<NAME>, etc.
_MAG_COL_RE = re.compile(
    r"^(magx|magy|magz|mag)_sensor_(\w+)$", re.IGNORECASE
)


def _discover_sensors(columns: list[str]) -> dict[str, dict[str, str]]:
    """
    Parse column names and return a mapping::

        { sensor_name: { "magx": col, "magy": col, "magz": col, "mag": col } }

    Supports two formats:

    * **Multi-sensor** — ``magx_sensor_UNO``, ``magy_sensor_UNO``, …
    * **Legacy single-sensor** — ``magx``, ``magy``, ``magz``, ``mag``
      (mapped to sensor name ``"default"``)
    """
    sensor_cols: dict[str, dict[str, str]] = {}

    for col in columns:
        m = _MAG_COL_RE.match(col)
        if m:
            component, name = m.group(1).lower(), m.group(2)
            sensor_cols.setdefault(name, {})[component] = col

    # Fall back to legacy column names if no suffixed columns were found.
    if not sensor_cols:
        legacy_present = [c for c in _LEGACY_MAG_COLS if c in columns]
        if legacy_present:
            sensor_cols["default"] = {c: c for c in _LEGACY_MAG_COLS if c in columns}

    return sensor_cols


def _validate(df: pd.DataFrame, file_path: str) -> dict[str, dict[str, str]]:
    """
    Validate the dataframe and return the discovered sensor column map.

    Raises
    ------
    ValueError
        On missing base columns, incomplete sensor column groups, partial
        optional groups, or non-finite numeric values.
    """
    # --- base required columns ----------------------------------------------
    missing_base = [c for c in _BASE_REQUIRED if c not in df.columns]
    if missing_base:
        raise ValueError(
            f"'{file_path}' is missing required columns: {missing_base}. "
            f"Found: {list(df.columns)}"
        )

    # --- sensor discovery ---------------------------------------------------
    sensor_map = _discover_sensors(list(df.columns))
    if not sensor_map:
        raise ValueError(
            f"'{file_path}' contains no recognisable magnetometer columns. "
            f"Expected either 'magx/magy/magz/mag' (legacy) or "
            f"'magx_sensor_<NAME>/...' (multi-sensor)."
        )

    # Each discovered sensor must have all four components.
    required_components = {"magx", "magy", "magz", "mag"}
    for name, cols in sensor_map.items():
        missing_comps = required_components - set(cols.keys())
        if missing_comps:
            raise ValueError(
                f"Sensor '{name}' in '{file_path}' is missing components "
                f"{missing_comps}. Found: {set(cols.keys())}"
            )

    # --- optional group completeness ----------------------------------------
    for group_name, group_cols in _OPTIONAL_GROUPS.items():
        present = [c for c in group_cols if c in df.columns]
        if 0 < len(present) < len(group_cols):
            missing_g = [c for c in group_cols if c not in df.columns]
            raise ValueError(
                f"'{file_path}' has a partial {group_name} group. "
                f"Expected all of {list(group_cols)} or none. Missing: {missing_g}"
            )

    # --- finite-value checks ------------------------------------------------
    numeric_cols: list[str] = [
        c for c in _BASE_REQUIRED if _BASE_REQUIRED[c] is not None
    ]
    for cols_dict in sensor_map.values():
        numeric_cols.extend(cols_dict.values())
    for group_cols in _OPTIONAL_GROUPS.values():
        numeric_cols.extend(c for c in group_cols if c in df.columns)

    for col in numeric_cols:
        arr = df[col].to_numpy()
        if not np.isfinite(arr).all():
            n_bad = int((~np.isfinite(arr)).sum())
            raise ValueError(
                f"Column '{col}' in '{file_path}' contains {n_bad} "
                f"non-finite value(s) (NaN or Inf)."
            )

    return sensor_map


def read_trajectory_advanced(file_path: str, DT: float) -> Trajectory:
    """
    Read a trajectory CSV and return a ``Trajectory`` with one
    ``MagneticSensor`` per detected sensor.

    Required base columns::

        timestamp, longitude, latitude, heading

    Magnetic columns — one of:

    * **Multi-sensor** ``magx_sensor_<NAME>``, ``magy_sensor_<NAME>``,
      ``magz_sensor_<NAME>``, ``mag_sensor_<NAME>`` (repeatable for each sensor)
    * **Legacy single-sensor** ``magx``, ``magy``, ``magz``, ``mag``

    Optional column groups (all-or-nothing)::

        ve, vn, vd          → velocity
        acc_x, acc_y, acc_z → acceleration

    Parameters
    ----------
    file_path : str
        Path to the CSV file.
    DT : float
        Sampling interval in seconds; timestamps are ``row_index * DT``.

    Returns
    -------
    Trajectory

    Examples
    --------
    >>> traj = read_trajectory("run_multi.csv", DT=0.1)
    >>> traj.sensor_names
    ['UNO', 'DUO']
    >>> traj.sensors["UNO"].field.shape
    (3, 1000)
    >>> traj.magnetic.shape   # (S, 3, N)
    (2, 3, 1000)
    """
    df = pd.read_csv(file_path)
    sensor_map = _validate(df, file_path)

    # Cast all numeric columns to float64 in one pass.
    all_numeric: list[str] = [
        c for c in _BASE_REQUIRED if _BASE_REQUIRED[c] is not None
    ]
    for cols_dict in sensor_map.values():
        all_numeric.extend(cols_dict.values())
    for group_cols in _OPTIONAL_GROUPS.values():
        all_numeric.extend(c for c in group_cols if c in df.columns)

    df[all_numeric] = df[all_numeric].astype(np.float64)

    # Build per-sensor MagneticSensor objects.
    sensors: dict[str, MagneticSensor] = {
        name: MagneticSensor(
            field=np.stack((
                df[cols["magx"]].to_numpy(),
                df[cols["magy"]].to_numpy(),
                df[cols["magz"]].to_numpy(),
            ), axis=0),
            norm=df[cols["mag"]].to_numpy(),
        )
        for name, cols in sensor_map.items()
    }

    has_velocity     = all(c in df.columns for c in _OPTIONAL_GROUPS["velocity"])
    has_acceleration = all(c in df.columns for c in _OPTIONAL_GROUPS["acceleration"])

    return Trajectory(
        timestamp    = df.index.to_numpy() * DT,
        longitude    = df["longitude"].to_numpy(),
        latitude     = df["latitude"].to_numpy(),
        heading      = df["heading"].to_numpy(),
        sensors      = sensors,
        velocity     = (
            np.stack((df["ve"], df["vn"], df["vd"]), axis=0)
            if has_velocity else None
        ),
        acceleration = (
            np.stack((df["acc_x"], df["acc_y"], df["acc_z"]), axis=0)
            if has_acceleration else None
        ),
    )