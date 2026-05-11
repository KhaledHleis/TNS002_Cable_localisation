"""
Cable Localisation — EKF Pipeline Tracker
==========================================
Implements the Extended Kalman Filter described in:

    Bharti, Lane & Wang (2020). "A Semi-Heuristic Approach for Tracking
    Buried Subsea Pipelines using Fluxgate Magnetometers."
    IEEE CASE 2020, pp. 469-475.

State vector
------------
    x_k = [x_k, y_k, ψ_k]^T

    x_k, y_k — pipe reference point in the vehicle Cartesian frame (m)
    ψ_k      — pipe orientation (rad), measured from east, CCW

Prediction model  (eq. 3–4)
----------------------------
    f(x_k) = [ x_k + d_k * cos(ψ_k),
               y_k + d_k * sin(ψ_k),
               ψ_k ]

    d_k = dot(ΔX_veh, V_k)   — vehicle displacement projected onto pipe axis
        ΔX_veh = vehicle_pos_k - vehicle_pos_{k-1}
        V_k    = [cos ψ_k, sin ψ_k]

    This is eq. 5:  d_k = <X_k, Ṽ_k>.  It slides the pipe reference point
    forward by however much the vehicle moved *along* the pipe direction.
    Using the raw vehicle step (not the pipe-to-vehicle vector) is the correct
    interpretation — the pipe point tracks along the cable at the vehicle's
    along-track pace.

Jacobian of f  (eq. 6)
-----------------------
    F_k = [[1, 0, -d_k * sin(ψ_k)],
           [0, 1,  d_k * cos(ψ_k)],
           [0, 0,  1             ]]

Observation model  (eq. 8–9)
-----------------------------
    z_k = h(x_k) + v_k = H x_k + v_k
    H   = [[1, 0, 0],
           [0, 1, 0]]
    z_k = vehicle_pos_k  (NOT [0,0]).

    The paper uses z_k = [0,0] because it works in the vehicle frame where
    the vehicle is always at the origin on a centre detection.  Here we work
    in a fixed Cartesian frame, so the observation is the actual vehicle
    position — which tells the filter where the pipe must be.

Observation noise  (eq. 10)
----------------------------
    R_k = max(1/|d_k|, 1) * σ²_v * I_{2×2}

    Inflates R when the vehicle is nearly stationary to avoid
    over-confident updates from tiny movements.
"""

from __future__ import annotations

import numpy as np


from helper.kalman import ExtendedKalmanFilter
from helper.utilities import latlon_to_cartesian

# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# EKF model functions
# ---------------------------------------------------------------------------

def _make_f(d_k: float):
    """
    Nonlinear prediction function f(x) for step d_k.
    Advances the pipe reference point along its own axis by d_k metres.
    """
    def f(x: np.ndarray) -> np.ndarray:
        px, py, psi = x
        return np.array([
            px + d_k * np.cos(psi),
            py + d_k * np.sin(psi),
            psi,
        ])
    return f


def _make_F_jacobian(d_k: float):
    """Jacobian of f w.r.t. state x  (eq. 6)."""
    def F_jac(x: np.ndarray) -> np.ndarray:
        _, _, psi = x
        return np.array([
            [1.0, 0.0, -d_k * np.sin(psi)],
            [0.0, 1.0,  d_k * np.cos(psi)],
            [0.0, 0.0,  1.0              ],
        ])
    return F_jac


# Observation: h(x) = H x  — we observe the pipe's (x,y) position directly.
# On a centre detection the vehicle is above the pipe, so z = vehicle_pos.
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
    EKF-based cable / pipeline localiser (Bharti et al. 2020).

    Parameters
    ----------
    sigma_v : float
        Observation noise std (m).  Paper uses 1.414.
    sigma_q : array_like, shape (3,)
        Process noise std for [x (m), y (m), ψ (rad)].
    P0_diag : array_like, shape (3,)
        Std of initial uncertainty for [x, y, ψ].
        Set P0_diag[2] large (e.g. π) — the paper notes ψ covariance must
        start high because the initial heading is unknown.
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

        # previous vehicle position — needed to compute d_k
        self._prev_veh: np.ndarray | None = None

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
        return self._ekf.x.copy() if self._ekf is not None else None

    @property
    def covariance(self) -> np.ndarray | None:
        return self._ekf.P.copy() if self._ekf is not None else None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialise(self, x0: float, y0: float, psi0: float = 0.0) -> None:
        """
        Initialise the filter at the first centre detection.

        Parameters
        ----------
        x0, y0 : float  — vehicle position at first detection (Cartesian, m)
                          This is also our first pipe position estimate.
        psi0   : float  — initial heading guess (rad).  Will converge quickly
                          once crossing detections arrive.
        """
        x_init = np.array([x0, y0, psi0])
        P_init = np.diag(self.P0_diag ** 2)
        Q      = np.diag(self.sigma_q ** 2)

        self._ekf = ExtendedKalmanFilter(
            x0         = x_init,
            P0         = P_init,
            f          = _make_f(0.0),
            h          = _h,
            F_jacobian = _make_F_jacobian(0.0),
            H_jacobian = _H_jacobian,
            Q          = Q,
            R          = np.eye(2) * self.sigma_v ** 2,
        )
        self._prev_veh = np.array([x0, y0])
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
        x_veh, y_veh        : float — current vehicle Cartesian position (m)
        is_centre_detection : bool  — True on a 'Centre' detection
        step_index          : int   — for bookkeeping

        Returns
        -------
        x_est : np.ndarray (3,)   — [x_pipe, y_pipe, ψ]
        P_est : np.ndarray (3, 3) — covariance
        """
        if not self._initialised:
            raise RuntimeError("Call initialise() before step().")

        veh = np.array([x_veh, y_veh])

        # ── d_k : vehicle displacement projected onto pipe axis  (eq. 5) ──
        # ΔX_veh = current vehicle pos – previous vehicle pos
        # d_k    = dot(ΔX_veh, [cos ψ, sin ψ])
        psi = self._ekf.x[2]
        V   = np.array([np.cos(psi), np.sin(psi)])
        d_k = float(np.dot(veh - self._prev_veh, V))
        self._prev_veh = veh.copy()

        # ── update EKF equations for this d_k ─────────────────────────────
        self._ekf.set_equations(
            f          = _make_f(d_k),
            h          = _h,
            F_jacobian = _make_F_jacobian(d_k),
            H_jacobian = _H_jacobian,
        )

        # ── predict ───────────────────────────────────────────────────────
        self._ekf.predict()

        # ── update on centre detections only  (Algorithm 2, line 3–8) ────
        if is_centre_detection:
            # R_k inflates when vehicle barely moves  (eq. 10)
            scale = max(1.0 / max(abs(d_k), 1e-6), 1.0)
            self._ekf.R = scale * self.sigma_v ** 2 * np.eye(2)

            # Observation = vehicle position (pipe is directly below)
            self._ekf.update(veh)
            self.updated_steps.append(step_index)

        self.state_history.append(self._ekf.x.copy())
        self.cov_history.append(self._ekf.P.copy())
        return self._ekf.x.copy(), self._ekf.P.copy()

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def get_pipe_estimate(self) -> tuple[float, float, float, float]:
        """Return (x, y, heading_deg, heading_rad) of the current estimate."""
        if not self._initialised:
            raise RuntimeError("Filter not initialised.")
        x, y, psi = self._ekf.x
        return float(x), float(y), float(np.rad2deg(psi) % 360), float(psi)


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
    psi0                : initial pipe heading guess (rad)

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