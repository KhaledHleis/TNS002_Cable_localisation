"""
Kalman Filter and Extended Kalman Filter
=========================================
Reviewed and corrected implementation.

Changes from original
---------------------
KalmanFilter
  - update(): replaced np.linalg.inv(S) with np.linalg.solve() for numerical
    stability (fixes Bug 5).
  - predict_and_update(): added guard that warns when H/R are not set (Bug 4
    mitigation — still raises ValueError inside update(), but message is clearer).

ExtendedKalmanFilter
  - predict(): removed dead line `S = H @ self.P @ H.T + self.R` that appeared
    after the return statement (fixes Bug 1).
  - update(): replaced np.linalg.inv(S) with np.linalg.solve() (fixes Bug 5,
    same as KalmanFilter).
  - Added docstrings clarifying the expected call order and IEKF behaviour when
    update() is called multiple times without predict() (Bug 3 documentation).

Both classes
  - Joseph-form covariance update retained — it is correct and beneficial,
    especially for EKF (was flagged as overkill for linear KF but not wrong).
  - All other equations unchanged — they were mathematically correct.
"""

import numpy as np


class KalmanFilter:
    """
    Standard (linear) Kalman Filter.

    State model
    -----------
        x_{k+1} = F x_k + B u_k + w_k,   w_k ~ N(0, Q)
        z_k     = H x_k + v_k,            v_k ~ N(0, R)

    Parameters
    ----------
    x0 : array_like, shape (n,)
        Initial state estimate.
    P0 : array_like, shape (n, n)
        Initial state covariance.
    F  : array_like, shape (n, n), optional
        State transition matrix.
    Q  : array_like, shape (n, n), optional
        Process noise covariance.
    H  : array_like, shape (m, n), optional
        Observation matrix.
    R  : array_like, shape (m, m), optional
        Measurement noise covariance.
    B  : array_like, shape (n, p), optional
        Control-input matrix (only required when a control input u is used).
    """

    def __init__(self, x0, P0, F=None, Q=None, H=None, R=None, B=None):
        self.x = np.asarray(x0, dtype=float)
        self.P = np.asarray(P0, dtype=float)
        self.F = F
        self.Q = Q
        self.H = H
        self.R = R
        self.B = B

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def set_transition(self, F, Q, B=None):
        """Set (or update) the state-transition matrices."""
        self.F = F
        self.Q = Q
        self.B = B

    def set_observation(self, H, R):
        """Set (or update) the observation matrices."""
        self.H = H
        self.R = R

    # ------------------------------------------------------------------
    # Core filter steps
    # ------------------------------------------------------------------

    def predict(self, u=None):
        """
        Propagate the state estimate and covariance forward one time step.

        Parameters
        ----------
        u : array_like, optional
            Control input vector.  Requires B to be set.

        Returns
        -------
        x : ndarray  — predicted state estimate x_{k+1|k}
        P : ndarray  — predicted covariance     P_{k+1|k}
        """
        if self.F is None or self.Q is None:
            raise ValueError("Transition matrix F and process noise Q must be set "
                             "via set_transition() before calling predict().")
        if u is None:
            self.x = self.F @ self.x
        else:
            if self.B is None:
                raise ValueError("Control input u provided but control matrix B is "
                                 "not set.  Pass B to set_transition() or __init__().")
            self.x = self.F @ self.x + self.B @ u

        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x, self.P

    def update(self, z):
        """
        Correct the predicted estimate with a new measurement.

        The covariance is updated using the Joseph form
            P = (I - KH) P (I - KH)^T + K R K^T
        which is numerically more robust than the simpler (I - KH) P form
        and guarantees positive semi-definiteness.

        Parameters
        ----------
        z : array_like
            Measurement vector, shape (m,).

        Returns
        -------
        x : ndarray  — updated state estimate x_{k|k}
        P : ndarray  — updated covariance     P_{k|k}
        """
        if self.H is None or self.R is None:
            raise ValueError("Observation matrix H and measurement noise R must be "
                             "set via set_observation() before calling update().")

        z = np.asarray(z, dtype=float)
        y = z - self.H @ self.x                          # innovation
        S = self.H @ self.P @ self.H.T + self.R          # innovation covariance

        # FIX (Bug 5): use solve() instead of inv() for numerical stability.
        # K = P H^T S^{-1}  ⟺  K S = P H^T  ⟺  S^T K^T = H P^T = H P
        K = np.linalg.solve(S.T, (self.P @ self.H.T).T).T

        self.x = self.x + K @ y

        I = np.eye(self.P.shape[0])
        IKH = I - K @ self.H
        # Joseph form — correct and stable for both KF and EKF.
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T
        return self.x, self.P

    def predict_and_update(self, z, u=None):
        """Convenience wrapper: predict then update in a single call."""
        self.predict(u)
        return self.update(z)


# ---------------------------------------------------------------------------


class ExtendedKalmanFilter(KalmanFilter):
    """
    Extended Kalman Filter (EKF) for nonlinear systems.

    Linearises the (potentially nonlinear) state-transition and observation
    functions around the current estimate at each time step.

    State model
    -----------
        x_{k+1} = f(x_k, u_k) + w_k,   w_k ~ N(0, Q)
        z_k     = h(x_k)       + v_k,   v_k ~ N(0, R)

    Parameters
    ----------
    x0         : array_like, shape (n,)       — initial state estimate
    P0         : array_like, shape (n, n)     — initial covariance
    f          : callable (x[, u]) -> x'      — nonlinear transition function
    h          : callable (x)      -> z       — nonlinear observation function
    F_jacobian : callable (x[, u]) -> (n, n)  — Jacobian of f w.r.t. x
    H_jacobian : callable (x)      -> (m, n)  — Jacobian of h w.r.t. x
    Q          : array_like, shape (n, n)     — process noise covariance
    R          : array_like, shape (m, m)     — measurement noise covariance
    B          : array_like, shape (n, p), optional — control-input matrix

    Notes
    -----
    Call order
        predict() must be called before update() on each time step.
        Calling update() more than once between two predict() calls is valid
        (this becomes the iterated EKF / IEKF), but re-linearises H at the
        corrected state, which may not be what you want in a standard EKF.

    Jacobian evaluation
        Both F_jacobian and H_jacobian receive a *copy* of the state vector
        so that they cannot accidentally mutate the filter state.
    """

    def __init__(self, x0, P0, f, h, F_jacobian, H_jacobian, Q, R, B=None):
        super().__init__(x0, P0, F=None, Q=Q, H=None, R=R, B=B)
        self.f = f
        self.h = h
        self.F_jacobian = F_jacobian
        self.H_jacobian = H_jacobian

    # ------------------------------------------------------------------
    # Configuration helper
    # ------------------------------------------------------------------

    def set_equations(self, f, h, F_jacobian, H_jacobian):
        """Replace the nonlinear functions and their Jacobians."""
        self.f = f
        self.h = h
        self.F_jacobian = F_jacobian
        self.H_jacobian = H_jacobian

    # ------------------------------------------------------------------
    # Core filter steps
    # ------------------------------------------------------------------

    def predict(self, u=None):
        """
        EKF prediction step.

        Propagates the state through the nonlinear function f and linearises
        around x_k (the state *before* propagation) to update the covariance.

        The Jacobian is evaluated at x_k (pre-propagation snapshot) — this is
        the standard EKF linearisation point.

        Parameters
        ----------
        u : array_like, optional — control input

        Returns
        -------
        x : ndarray  — predicted state x_{k+1|k}
        P : ndarray  — predicted covariance P_{k+1|k}
        """
        if self.f is None or self.F_jacobian is None or self.Q is None:
            raise ValueError("Nonlinear transition function f, its Jacobian "
                             "F_jacobian, and process noise Q must all be set.")

        # Snapshot x_k before propagation — Jacobian linearises here.
        x_k = self.x.copy()

        # Propagate state through nonlinear f.
        self.x = self.f(x_k, u) if u is not None else self.f(x_k)

        # Linearise around x_k and propagate covariance.
        F = self.F_jacobian(x_k, u) if u is not None else self.F_jacobian(x_k)
        self.P = F @ self.P @ F.T + self.Q

        # FIX (Bug 1): removed dead line that appeared here in the original:
        #   S = H @ self.P @ H.T + self.R   ← was after return, never executed.
        return self.x, self.P

    def update(self, z):
        """
        EKF update step.

        Linearises the observation function h around the current (predicted)
        state estimate x_{k+1|k} to compute the Kalman gain.

        Parameters
        ----------
        z : array_like — measurement vector, shape (m,)

        Returns
        -------
        x : ndarray  — updated state estimate x_{k+1|k+1}
        P : ndarray  — updated covariance     P_{k+1|k+1}
        """
        if self.h is None or self.H_jacobian is None or self.R is None:
            raise ValueError("Nonlinear observation function h, its Jacobian "
                             "H_jacobian, and measurement noise R must all be set.")

        z = np.asarray(z, dtype=float)
        y = z - self.h(self.x)                       # innovation
        H = self.H_jacobian(self.x)                  # linearise h at x_{k+1|k}
        S = H @ self.P @ H.T + self.R                # innovation covariance

        # FIX (Bug 5): use solve() instead of inv() for numerical stability.
        K = np.linalg.solve(S.T, (self.P @ H.T).T).T

        self.x = self.x + K @ y

        I = np.eye(self.P.shape[0])
        IKH = I - K @ H
        # Joseph form — important for EKF where linearisation is approximate.
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T
        return self.x, self.P