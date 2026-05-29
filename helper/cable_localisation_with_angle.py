"""
Cable Localisation — EKF Pipeline Tracker  (Cable-Frame + Angle Observation)
=============================================================================
Extends the cable-frame EKF (Bharti et al. 2020) with a third observation
component: the cable heading angle α derived from dual-sensor time-difference
detections.

Reference-frame convention
--------------------------
State vector:

    x_k = [dx_k, dy_k, ψ_k]^T

    dx_k, dy_k — vehicle offset from cable in world Cartesian (m)
                 On a centre detection both are driven toward zero.
    ψ_k        — cable orientation (rad), East=0, CCW.

Prediction model  (cable frame — cable is fixed at origin)
----------------------------------------------------------
    f(x_k) = [ dx_k + Δx_veh,
               dy_k + Δy_veh,
               ψ_k            ]

    Jacobian F_k = I_{3×3}   (f is linear in x)

Observation modes
-----------------
Three cases depending on what is available at step k:

  A) Centre detection + angle observation
     z_k = [0, 0, ψ_obs_k]^T
     H   = I_{3×3}
     R_k = diag(scale·σ²_v, scale·σ²_v, σ²_α)

  B) Centre detection only  (no paired dual-sensor angle)
     z_k = [0, 0]^T
     H   = [[1,0,0],[0,1,0]]
     R_k = scale·σ²_v · I_{2×2}

  C) Angle observation only  (no centre detection)
     z_k = [ψ_obs_k]
     H   = [[0, 0, 1]]
     R_k = [σ²_α]

Angle observation  ψ_obs
------------------------
    ψ_obs = resolve_ambiguity( φ_k - α_k )

    φ_k  — drone heading at step k  (from mag signal, radians)
    α_k  — incidence angle from dual-sensor time diff (radians)

    The 180° ambiguity is resolved by picking the candidate closest to
    the current ψ estimate.

Observation noise
-----------------
    σ_v  — position observation std (m)
    σ_α  — angle observation std (rad); tune based on sensor separation
            and velocity estimation quality.
    scale = max(1/|d_k|, 1)   (inflates R when vehicle barely moves)
"""

from __future__ import annotations

import numpy as np

from helper.kalman import ExtendedKalmanFilter
from helper.utilities import latlon_to_cartesian
from helper.heading_estimator import resolve_heading_ambiguity


# ---------------------------------------------------------------------------
# EKF model functions
# ---------------------------------------------------------------------------

def _make_f(delta_veh: np.ndarray):
    """Prediction in cable frame: vehicle offset grows by world-frame step."""
    def f(x: np.ndarray) -> np.ndarray:
        dx, dy, psi = x
        return np.array([dx + delta_veh[0], dy + delta_veh[1], psi])
    return f


def _make_F_jacobian(_delta_veh: np.ndarray):
    """Jacobian of f — identity (f is linear in x)."""
    def F_jac(_x: np.ndarray) -> np.ndarray:
        return np.eye(3)
    return F_jac


# --- observation matrices for the three cases ---

_H_FULL  = np.eye(3)                          # case A: position + angle
_H_POS   = np.array([[1.,0.,0.],[0.,1.,0.]])  # case B: position only
_H_ANGLE = np.array([[0., 0., 1.]])           # case C: angle only


def _h_full(x):  return _H_FULL  @ x
def _h_pos(x):   return _H_POS   @ x
def _h_angle(x): return _H_ANGLE @ x

def _Hj_full(_x):  return _H_FULL.copy()
def _Hj_pos(_x):   return _H_POS.copy()
def _Hj_angle(_x): return _H_ANGLE.copy()


# ---------------------------------------------------------------------------
# CableLocaliser
# ---------------------------------------------------------------------------

class CableLocaliser:
    """
    EKF cable localiser — cable frame with optional angle observations.

    Parameters
    ----------
    sigma_v : float
        Position observation noise std (m).
    sigma_alpha : float
        Angle observation noise std (rad).  Tune to dual-sensor geometry
        and velocity estimation quality.  Default π/12 (≈15°).
    sigma_q : array_like (3,)
        Process noise std  [dx(m), dy(m), ψ(rad)].
    P0_diag : array_like (3,)
        Initial uncertainty std  [dx(m), dy(m), ψ(rad)].
        Keep P0_diag[2] large — heading is unknown at start.
    """

    def __init__(
        self,
        sigma_v:     float = 1.414,
        sigma_alpha: float = np.deg2rad(15.0),
        sigma_q:     np.ndarray = np.array([0.1, 0.1, np.deg2rad(0.001)]),  # cable is static — ψ barely drifts
        P0_diag:     np.ndarray = np.array([2.0, 2.0, np.pi]),
    ):
        self.sigma_v     = float(sigma_v)
        self.sigma_alpha = float(sigma_alpha)
        self.sigma_q     = np.asarray(sigma_q,  dtype=float)
        self.P0_diag     = np.asarray(P0_diag,  dtype=float)

        self._ekf: ExtendedKalmanFilter | None = None
        self._initialised = False
        self._prev_veh:    np.ndarray | None = None

        # histories
        self.state_history:    list[np.ndarray] = []
        self.cov_history:      list[np.ndarray] = []
        self.updated_steps:    list[int]        = []   # centre-detection steps
        self.angle_obs_steps:  list[int]        = []   # angle-observation steps

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

    @property
    def cable_world_position(self) -> np.ndarray | None:
        """Current cable position in world Cartesian: vehicle_world - state[:2]."""
        if self._ekf is None or self._prev_veh is None:
            return None
        return self._prev_veh - self._ekf.x[:2]

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialise(
        self,
        x0:   float,
        y0:   float,
        psi0: float = 0.0,
    ) -> None:
        """
        Initialise at the first centre detection.

        Vehicle is directly above the cable → offset (dx, dy) = (0, 0).

        Parameters
        ----------
        x0, y0 : float  — world Cartesian vehicle position (m)
        psi0   : float  — initial cable heading guess (rad)
        """
        x_init = np.array([0.0, 0.0, psi0])
        P_init = np.diag(self.P0_diag ** 2)
        Q      = np.diag(self.sigma_q  ** 2)

        self._ekf = ExtendedKalmanFilter(
            x0         = x_init,
            P0         = P_init,
            f          = _make_f(np.zeros(2)),
            h          = _h_pos,
            F_jacobian = _make_F_jacobian(np.zeros(2)),
            H_jacobian = _Hj_pos,
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
        x_veh:               float,
        y_veh:               float,
        is_centre_detection: bool,
        drone_heading:       float | None = None,
        alpha_rad:           float | None = None,
        step_index:          int   = -1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        One EKF cycle: predict → update.

        The update depends on what observations are available:
          - centre + angle  → 3-component update  z = [0, 0, ψ_obs]
          - centre only     → 2-component update  z = [0, 0]
          - angle only      → 1-component update  z = [ψ_obs]
          - nothing         → predict only

        Parameters
        ----------
        x_veh, y_veh        : float        — current vehicle world position (m)
        is_centre_detection : bool         — True on a centre detection
        drone_heading       : float | None — drone heading φ_k (rad); needed
                                             for angle observation
        alpha_rad           : float | None — incidence angle α_k (rad) from
                                             dual-sensor estimate
        step_index          : int          — bookkeeping

        Returns
        -------
        x_est : np.ndarray (3,)    [dx, dy, ψ]
        P_est : np.ndarray (3, 3)
        """
        if not self._initialised:
            raise RuntimeError("Call initialise() before step().")

        veh       = np.array([x_veh, y_veh])
        delta_veh = veh - self._prev_veh
        self._prev_veh = veh.copy()

        # along-cable displacement (for R inflation)
        psi = self._ekf.x[2]
        V   = np.array([np.cos(psi), np.sin(psi)])
        d_k = float(np.dot(delta_veh, V))
        scale = min(max(1.0 / max(abs(d_k), 1e-6), 1.0), 100.0)  # cap: don't freeze on turnarounds

        # ── update prediction equations ───────────────────────────────
        self._ekf.set_equations(
            f          = _make_f(delta_veh),
            h          = _h_pos,           # placeholder; overridden below
            F_jacobian = _make_F_jacobian(delta_veh),
            H_jacobian = _Hj_pos,
        )

        # ── predict ───────────────────────────────────────────────────
        self._ekf.predict()

        # ── build observation ─────────────────────────────────────────
        has_angle = (drone_heading is not None) and (alpha_rad is not None)
        psi_obs: float | None = None

        if has_angle:
            raw_psi_obs = drone_heading - alpha_rad
            psi_obs     = resolve_heading_ambiguity(raw_psi_obs, self._ekf.x[2], alpha_rad=alpha_rad)
            # Diagnostic: first 3 angle observations
            if len(self.angle_obs_steps) < 3:
                print(f"  [angle obs #{len(self.angle_obs_steps)+1}] "
                      f"phi={np.rad2deg(drone_heading):.1f}deg  "
                      f"alpha={np.rad2deg(alpha_rad):.1f}deg  "
                      f"raw_psi={np.rad2deg(raw_psi_obs):.1f}deg  "
                      f"resolved={np.rad2deg(psi_obs):.1f}deg  "
                      f"psi_current={np.rad2deg(self._ekf.x[2]):.1f}deg")
            self.angle_obs_steps.append(step_index)

        if is_centre_detection and has_angle:
            # ── Case A: position + angle ──────────────────────────────
            self._ekf.set_equations(
                f          = _make_f(delta_veh),
                h          = _h_full,
                F_jacobian = _make_F_jacobian(delta_veh),
                H_jacobian = _Hj_full,
            )
            self._ekf.R = np.diag([
                scale * self.sigma_v     ** 2,
                scale * self.sigma_v     ** 2,
                       self.sigma_alpha  ** 2,
            ])
            self._ekf.update(np.array([0.0, 0.0, psi_obs]))
            self.updated_steps.append(step_index)

        elif is_centre_detection:
            # ── Case B: position only ─────────────────────────────────
            self._ekf.set_equations(
                f          = _make_f(delta_veh),
                h          = _h_pos,
                F_jacobian = _make_F_jacobian(delta_veh),
                H_jacobian = _Hj_pos,
            )
            self._ekf.R = scale * self.sigma_v ** 2 * np.eye(2)
            self._ekf.update(np.zeros(2))
            self.updated_steps.append(step_index)

        elif has_angle:
            # ── Case C: angle only ────────────────────────────────────
            self._ekf.set_equations(
                f          = _make_f(delta_veh),
                h          = _h_angle,
                F_jacobian = _make_F_jacobian(delta_veh),
                H_jacobian = _Hj_angle,
            )
            self._ekf.R = np.array([[self.sigma_alpha ** 2]])
            self._ekf.update(np.array([psi_obs]))

        # ── store ─────────────────────────────────────────────────────
        self.state_history.append(self._ekf.x.copy())
        self.cov_history.append(self._ekf.P.copy())
        return self._ekf.x.copy(), self._ekf.P.copy()

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def get_pipe_estimate(self) -> tuple[float, float, float, float]:
        """
        Return (cable_x, cable_y, heading_deg, heading_rad) in world Cartesian.
        cable_pos = vehicle_world - state[:2]
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
    drone_headings: np.ndarray | None = None,
    alpha_per_step: np.ndarray | None = None,
    sigma_v:        float = 1.414,
    sigma_alpha:    float = np.deg2rad(15.0),
    psi0:           float = 0.0,
    sigma_q:        np.ndarray | None = None,
) -> tuple[CableLocaliser, np.ndarray, np.ndarray]:
    """
    Convert trajectory to Cartesian, then run the full EKF localiser.

    Parameters
    ----------
    longitude, latitude : (N,)  — decimal degrees
    detection_mask      : (N,) bool — True on centre detections
    drone_headings      : (N,) float | None — drone heading φ (rad) per step
    alpha_per_step      : (N,) float | None — incidence angle α (rad) per step;
                          NaN where no angle estimate is available
    sigma_v             : position observation noise std (m)
    sigma_alpha         : angle observation noise std (rad)
    psi0                : initial cable heading guess (rad)

    Returns
    -------
    localiser, x_cart, y_cart
    """
    x_cart, y_cart = latlon_to_cartesian(longitude, latitude)

    first_det = int(np.argmax(detection_mask))
    if not detection_mask[first_det]:
        raise ValueError("No centre detection in detection_mask.")

    _sq = sigma_q if sigma_q is not None else np.array([0.1, 0.1, np.deg2rad(0.001)])
    localiser = CableLocaliser(sigma_v=sigma_v, sigma_alpha=sigma_alpha, sigma_q=_sq)
    localiser.initialise(x_cart[first_det], y_cart[first_det], psi0)

    for k in range(first_det + 1, len(x_cart)):
        phi_k = float(drone_headings[k]) if drone_headings is not None else None
        alpha_k = None
        if alpha_per_step is not None:
            v = float(alpha_per_step[k])
            if not np.isnan(v):
                alpha_k = v

        localiser.step(
            x_veh               = float(x_cart[k]),
            y_veh               = float(y_cart[k]),
            is_centre_detection = bool(detection_mask[k]),
            drone_heading       = phi_k,
            alpha_rad           = alpha_k,
            step_index          = k,
        )

    return localiser, x_cart, y_cart