import numpy as np
from helper.kalman import KalmanFilter, ExtendedKalmanFilter
# ── paste your class definitions here, or import them ──────────────────────────
# from kalman_filter import KalmanFilter, ExtendedKalmanFilter


# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 1 – KalmanFilter: tracking a 1-D object moving at constant velocity
# ══════════════════════════════════════════════════════════════════════════════
# State vector:  x = [position, velocity]ᵀ
# Observation:   z = [position]            (noisy GPS reading)

dt = 0.1          # time-step (s)
n_steps = 50

# --- matrices -----------------------------------------------------------------
F = np.array([[1, dt],   # state transition:  pos += vel * dt
              [0,  1]])

Q = np.array([[1e-4, 0],   # small process noise
              [0,   1e-4]])

H = np.array([[1, 0]])     # we only observe position

R = np.array([[1.0]])      # measurement noise variance

# --- initial state & covariance -----------------------------------------------
x0 = np.array([0.0, 1.0])   # starts at pos=0, vel=1 m/s
P0 = np.eye(2) * 0.5

kf = KalmanFilter(x0, P0)
kf.set_transition(F, Q)
kf.set_observation(H, R)

# --- simulate & filter --------------------------------------------------------
true_positions = []
filtered_positions = []

true_pos, true_vel = 0.0, 1.0
for _ in range(n_steps):
    true_pos += true_vel * dt
    noisy_meas = np.array([true_pos + np.random.normal(0, 1.0)])

    x_est, _ = kf.predict_and_update(noisy_meas)

    true_positions.append(true_pos)
    filtered_positions.append(x_est[0])

print("=== KalmanFilter: constant-velocity tracking ===")
print(f"True final position  : {true_positions[-1]:.3f} m")
print(f"Filter final estimate: {filtered_positions[-1]:.3f} m")
print()


# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 2 – KalmanFilter with control input: robot with known acceleration
# ══════════════════════════════════════════════════════════════════════════════
# State:   x = [position, velocity]ᵀ
# Control: u = [acceleration]

B = np.array([[0.5 * dt**2],   # control-input matrix
              [dt]])

kf2 = KalmanFilter(x0.copy(), P0.copy())
kf2.set_transition(F, Q, B=B)
kf2.set_observation(H, R)

u = np.array([0.5])   # constant 0.5 m/s² acceleration

print("=== KalmanFilter: with control input ===")
for step in range(5):
    noisy_meas = np.array([step * dt + 0.5 * 0.5 * (step * dt)**2
                           + np.random.normal(0, 0.5)])
    x_est, P_est = kf2.predict_and_update(noisy_meas, u=u)
    print(f"  step {step+1}: pos={x_est[0]:.3f} m, vel={x_est[1]:.3f} m/s")
print()


# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 3 – ExtendedKalmanFilter: tracking a pendulum (nonlinear dynamics)
# ══════════════════════════════════════════════════════════════════════════════
# State:   x = [angle θ, angular-velocity ω]ᵀ
# Dynamics (continuous, Euler-discretised):
#   θ_{k+1} = θ_k + ω_k * dt
#   ω_{k+1} = ω_k - (g/L) * sin(θ_k) * dt
# Observation: z = sin(θ)   (nonlinear!)

g, L = 9.81, 1.0   # gravity, pendulum length
dt_ekf = 0.05


def f_pendulum(x, u=None):
    theta, omega = x
    theta_next = theta + omega * dt_ekf
    omega_next  = omega - (g / L) * np.sin(theta) * dt_ekf
    return np.array([theta_next, omega_next])


def F_jacobian(x, u=None):
    theta, _ = x
    return np.array([[1,             dt_ekf],
                     [-(g/L)*np.cos(theta)*dt_ekf, 1]])


def h_pendulum(x):
    return np.array([np.sin(x[0])])


def H_jacobian(x):
    return np.array([[np.cos(x[0]), 0]])


Q_ekf = np.diag([1e-5, 1e-4])
R_ekf = np.array([[0.05]])

x0_ekf = np.array([0.5, 0.0])   # start at 0.5 rad, at rest
P0_ekf = np.diag([0.1, 0.1])

ekf = ExtendedKalmanFilter(
    x0_ekf, P0_ekf,
    f=f_pendulum, h=h_pendulum,
    F_jacobian=F_jacobian, H_jacobian=H_jacobian,
    Q=Q_ekf, R=R_ekf
)

print("=== ExtendedKalmanFilter: pendulum tracking ===")
true_state = x0_ekf.copy()
for step in range(6):
    # ground-truth step
    true_state = f_pendulum(true_state)
    z = np.array([np.sin(true_state[0]) + np.random.normal(0, np.sqrt(0.05))])

    x_est, _ = ekf.predict_and_update(z)
    print(f"  step {step+1}: true θ={true_state[0]:.3f} rad | "
          f"est. θ={x_est[0]:.3f} rad, ω={x_est[1]:.3f} rad/s")