"""
Cable Localisation — EKF Pipeline Tracker  (Cable-Frame Edition)
=================================================================
Implements the Extended Kalman Filter described in:

    Bharti, Lane & Wang (2020). "A Semi-Heuristic Approach for Tracking
    Buried Subsea Pipelines using Fluxgate Magnetometers."
    IEEE CASE 2020, pp. 469-475.

Reference-frame convention
--------------------------
The state is expressed in the **cable frame**: the cable is always at the
origin.  The state vector represents the vehicle's offset from the cable:

    x_k = [dx_k, dy_k, ψ_k]^T

    dx_k, dy_k — vehicle position relative to the cable (m)
                 On a perfect centre detection both are driven to zero.
    ψ_k        — cable orientation (rad), measured from east, CCW.

This matches the original Bharti et al. paper exactly: the observation on a
centre detection is z_k = [0, 0] (vehicle is directly above the cable).

Prediction model
----------------
The vehicle moves by ΔX_veh = [Δx, Δy] in world coordinates each step.
Since the cable does not move, the vehicle's offset from the cable changes
by the same world-frame displacement:

    f(x_k) = [ dx_k + Δx,
               dy_k + Δy,
               ψ_k        ]

Jacobian of f
-------------
f is linear in x, so the Jacobian is simply the identity:

    F_k = I_{3×3}

Observation model  (eq. 8–9)
-----------------------------
    z_k = H x_k + v_k
    H   = [[1, 0, 0],
           [0, 1, 0]]

On a centre detection the vehicle is above the cable, so:

    z_k = [0, 0]    ← vehicle offset from cable is (ideally) zero

Observation noise  (eq. 10)
----------------------------
    R_k = max(1 / |d_k|, 1) * σ²_v * I_{2×2}

    d_k = dot(ΔX_veh, [cos ψ, sin ψ])   — along-cable displacement

    Inflates R when the vehicle is nearly stationary to avoid
    over-confident updates from tiny movements.
"""

from __future__ import annotations

import numpy as np

from helper.kalman import ExtendedKalmanFilter
from helper.utilities import latlon_to_cartesian


# ---------------------------------------------------------------------------
# EKF model functions
# ---------------------------------------------------------------------------

def _make_f(delta_veh: np.ndarray):
    """
    Prediction: vehicle offset from cable = previous offset + world-frame step.

    Parameters
    ----------
    delta_veh : np.ndarray, shape (2,)
        ΔX_veh = current_vehicle_pos - prev_vehicle_pos  (world Cartesian, m)
    """
    def f(x: np.ndarray) -> np.ndarray:
        dx, dy, psi = x
        return np.array([
            dx + delta_veh[0],
            dy + delta_veh[1],
            psi,
        ])
    return f


def _make_F_jacobian(_delta_veh: np.ndarray):
    """
    Jacobian of f w.r.t. state x.

    f is linear in x (the state does not appear inside a trig function),
    so F is the constant identity matrix regardless of the current state or
    vehicle step.
    """
    def F_jac(_x: np.ndarray) -> np.ndarray:
        return np.eye(3)
    return F_jac


# Observation matrix: we observe the vehicle's (dx, dy) offset directly.
_H_OBS = np.array([[1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0]])


def _h(x: np.ndarray) -> np.ndarray:
    return _H_OBS @ x


def _H_jacobian(_x: np.ndarray) -> np.ndarray:
    return _H_OBS.copy()


# ---------------------------------------------------------------------------
# CableLocaliser
# ---------------------------------------------------------------------------

class CableLocaliser:
    """
    EKF-based cable / pipeline localiser (Bharti et al. 2020) — cable frame.

    The state [dx, dy, ψ] represents the vehicle's offset from the cable and
    the cable's heading.  On a centre detection the EKF update drives (dx, dy)
    toward [0, 0].

    Parameters
    ----------
    sigma_v : float
        Observation noise std (m).  Paper default: 1.414.
    sigma_q : array_like, shape (3,)
        Process noise std for [dx (m), dy (m), ψ (rad)].
    P0_diag : array_like, shape (3,)
        Std of initial uncertainty for [dx, dy, ψ].
        P0_diag[2] should be large (e.g. π) — heading is unknown at start.
    """

    def __init__(
        self,
        sigma_v: float = 1.414,
        sigma_q: np.ndarray = np.array([0.1, 0.1, np.deg2rad(1.0)]),
        P0_diag: np.ndarray = np.array([2.0, 2.0, np.pi]),
    ):
        self.sigma_v = float(sigma_v)
        self.sigma_q = np.asarray(sigma_q, dtype=float)
        self.P0_diag = np.asarray(P0_diag, dtype=float)

        self._ekf: ExtendedKalmanFilter | None = None
        self._initialised = False

        # previous vehicle world position — needed to compute ΔX_veh
        self._prev_veh: np.ndarray | None = None

        # world-frame position of the cable (set at initialisation, fixed)
        self._cable_world: np.ndarray | None = None

        # histories
        self.state_history: list[np.ndarray] = []
        self.cov_history:   list[np.ndarray] = []
        self.updated_steps: list[int]        = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_initialised(self) -> bool:
        return self._initialised

    @property
    def state(self) -> np.ndarray | None:
        """Current state [dx, dy, ψ] in cable frame."""
        return self._ekf.x.copy() if self._ekf is not None else None

    @property
    def covariance(self) -> np.ndarray | None:
        return self._ekf.P.copy() if self._ekf is not None else None

    @property
    def cable_world_position(self) -> np.ndarray | None:
        """
        World-frame Cartesian position of the cable reference point (m).
        Derived as: cable_world = vehicle_world - state[:2].
        Updated every step.
        """
        if self._ekf is None or self._prev_veh is None:
            return None
        return self._prev_veh - self._ekf.x[:2]

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialise(self, x0: float, y0: float, psi0: float = 0.0) -> None:
        """
        Initialise the filter at the first centre detection.

        At initialisation the vehicle is directly above the cable, so the
        vehicle's offset from the cable is [0, 0].

        Parameters
        ----------
        x0, y0 : float  — vehicle world-Cartesian position at first detection (m)
        psi0   : float  — initial cable heading guess (rad).
                          Converges quickly once further detections arrive.
        """
        # Cable-frame state: vehicle starts at offset (0, 0) from cable.
        x_init = np.array([0.0, 0.0, psi0])
        P_init = np.diag(self.P0_diag ** 2)
        Q      = np.diag(self.sigma_q ** 2)

        self._ekf = ExtendedKalmanFilter(
            x0         = x_init,
            P0         = P_init,
            f          = _make_f(np.zeros(2)),
            h          = _h,
            F_jacobian = _make_F_jacobian(np.zeros(2)),
            H_jacobian = _H_jacobian,
            Q          = Q,
            R          = np.eye(2) * self.sigma_v ** 2,
        )
        self._prev_veh    = np.array([x0, y0])
        self._cable_world = np.array([x0, y0])   # cable at vehicle on first det.
        self._initialised = True

        self.state_history.append(x_init.copy())
        self.cov_history.append(P_init.copy())

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        x_veh: float,
        y_veh: float,
        is_centre_detection: bool,
        step_index: int = -1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        One EKF cycle: predict, then update if 'Centre' detection.

        Parameters
        ----------
        x_veh, y_veh        : float — current vehicle world-Cartesian position (m)
        is_centre_detection : bool  — True on a 'Centre' detection
        step_index          : int   — for bookkeeping

        Returns
        -------
        x_est : np.ndarray (3,)   — [dx, dy, ψ]  (vehicle offset from cable)
        P_est : np.ndarray (3, 3) — covariance
        """
        if not self._initialised:
            raise RuntimeError("Call initialise() before step().")

        veh       = np.array([x_veh, y_veh])
        delta_veh = veh - self._prev_veh          # world-frame vehicle step
        self._prev_veh = veh.copy()

        # ── d_k : along-cable component of ΔX_veh  (for R inflation) ─────
        psi = self._ekf.x[2]
        V   = np.array([np.cos(psi), np.sin(psi)])
        d_k = float(np.dot(delta_veh, V))

        # ── update EKF equations for this step ────────────────────────────
        self._ekf.set_equations(
            f          = _make_f(delta_veh),
            h          = _h,
            F_jacobian = _make_F_jacobian(delta_veh),
            H_jacobian = _H_jacobian,
        )

        # ── predict ───────────────────────────────────────────────────────
        self._ekf.predict()

        # ── update on centre detections only  (Algorithm 2, line 3–8) ────
        if is_centre_detection:
            # Inflate R when vehicle is nearly stationary  (eq. 10)
            scale = max(1.0 / max(abs(d_k), 1e-6), 1.0)
            self._ekf.R = scale * self.sigma_v ** 2 * np.eye(2)

            # z_k = [0, 0]: vehicle is directly above the cable
            self._ekf.update(np.zeros(2))
            self.updated_steps.append(step_index)

        self.state_history.append(self._ekf.x.copy())
        self.cov_history.append(self._ekf.P.copy())
        return self._ekf.x.copy(), self._ekf.P.copy()

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def get_pipe_estimate(self) -> tuple[float, float, float, float]:
        """
        Return (cable_x, cable_y, heading_deg, heading_rad) in world Cartesian.

        cable_x/y = vehicle_world - state[:2]
        """
        if not self._initialised:
            raise RuntimeError("Filter not initialised.")
        cable_pos = self.cable_world_position
        psi       = self._ekf.x[2]
        return (
            float(cable_pos[0]),
            float(cable_pos[1]),
            float(np.rad2deg(psi) % 360),
            float(psi),
        )


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def localise_cable(
    longitude:      np.ndarray,
    latitude:       np.ndarray,
    detection_mask: np.ndarray,
    sigma_v:        float = 1.414,
    psi0:           float = 0.0,
) -> tuple[CableLocaliser, np.ndarray, np.ndarray]:
    """
    Convert trajectory to Cartesian, then run the full EKF localiser.

    Parameters
    ----------
    longitude, latitude : shape (N,)  — decimal degrees
    detection_mask      : shape (N,) bool — True on centre detections
    sigma_v             : observation noise std (m)
    psi0                : initial cable heading guess (rad)

    Returns
    -------
    localiser : CableLocaliser
    x_cart    : np.ndarray (N,)  — vehicle east  (m)
    y_cart    : np.ndarray (N,)  — vehicle north (m)
    """
    x_cart, y_cart = latlon_to_cartesian(longitude, latitude)

    first_det = int(np.argmax(detection_mask))
    if not detection_mask[first_det]:
        raise ValueError("No centre detection in detection_mask — cannot initialise.")

    localiser = CableLocaliser(sigma_v=sigma_v)
    localiser.initialise(x_cart[first_det], y_cart[first_det], psi0)

    for k in range(first_det + 1, len(x_cart)):
        localiser.step(
            x_veh               = float(x_cart[k]),
            y_veh               = float(y_cart[k]),
            is_centre_detection = bool(detection_mask[k]),
            step_index          = k,
        )

    return localiser, x_cart, y_cart