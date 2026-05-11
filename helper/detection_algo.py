import numpy as np

# region #* detection helper functions


def __haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great-circle distance between two points on the Earth in meters.
    """
    R = 6371000  # Earth radius in meters
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def estimate_average_velocity(long, lat, time_interval=1.0):
    """
    Estimate average speed (scalar) in meters per second between consecutive GPS points.

    Parameters:
        long: list or np.array of longitudes
        lat: list or np.array of latitudes
        time_interval: time difference between consecutive points in seconds (default 1s)

    Returns:
        avg_velocity: average speed in m/s
    """
    long = np.array(long)
    lat = np.array(lat)

    # Calculate distances between consecutive points
    distances = __haversine(long[:-1], lat[:-1], long[1:], lat[1:])

    # Average velocity = total distance / total time
    total_distance = np.sum(distances)
    total_time = len(distances) * time_interval

    avg_velocity = total_distance / total_time if total_time > 0 else 0
    return avg_velocity


# endregion

# region #* basis functions


def __anderson_basis_functions(N, dov: float = 1):
    """Returns orthogonalized Anderson basis functions using Gram–Schmidt

    Args:
        N (int): number of samples
        nbf (int): number of basis functions
        dov (float): domain scaling
    """
    basis = np.zeros((3, N))
    Tau = np.linspace(-100, 100, N)
    w = (1 + Tau**2) ** (5 / 2)

    # original Anderson basis
    for i in range(3):
        if i == 0:
            basis[i] = np.sqrt(128 / (35 * np.pi)) / w
        elif i == 1:
            basis[i] = np.sqrt(128 / (5 * np.pi)) * Tau / w
        elif i == 2:
            basis[i] = np.sqrt(56 / np.pi) * (Tau**2 - 1 / 7) / w

    return basis, dov * Tau / 2


def __cable_basis_functions(N, dov: float = 1, theta: float = 90):
    """Returns orthogonalized cable basis functions using Gram–Schmidt

    Args:
        N (int): number of samples
        nbf (int): number of basis functions
        dov (float): domain scaling (velocity over CPA distance)
        theta (float): angle of incidence
    """
    basis = np.zeros((2, N))
    Tau = np.linspace(-100, 100, N)
    oos = 1 / np.sin(np.deg2rad(theta))
    t = dov * oos * Tau
    w = 1 + t**2

    preterm = np.sqrt(2 / np.pi)
    basis[0] = preterm * 1 / w
    basis[1] = preterm * t / w

    return basis, t


def __gradient_basis_functions(N, dov: float = 1, theta: float = 90, l: float = 1):
    """Returns orthogonalized gradient basis functions using Gram–Schmidt

    Args:
        N (int): number of samples
        nbf (int): number of basis functions
        dov (float): domain scaling (velocity over CPA distance)
        theta (float): angle of incidence
        l (float): distance between the two gradient sensors (e.g., magnetometers)
    """
    basis = np.zeros((2, N))
    Tau = np.linspace(-100, 100, N)
    oos = 1 / np.sin(np.deg2rad(theta))
    t = dov * oos * Tau
    w = (1 + t**2) ** 2

    preterm = np.sqrt(4 / np.pi)
    basis[0] = -preterm * 2 * t / w
    basis[1] = preterm * (1 - t**2) / w

    return basis, t


# endregion


# region #* detection algorithm
def __Matched_filter(pulse_shape, rx_channel):
    conv = np.convolve(rx_channel, pulse_shape, mode="same")

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(pulse_shape, label=f"Pulse shape (len={len(pulse_shape)})")
    # plt.plot(conv, label="Matched filter output")
    # plt.xlabel("Time")
    # plt.ylabel("Amplitude")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    return conv


def detect_anomalies(
    signal: np.ndarray,
    time: np.ndarray,
    filter_width: int = 5000,
    drone_velocity: float = 1.0,
    method: str = "AMF",
) -> tuple[np.ndarray, np.ndarray]:
    signal = np.atleast_2d(signal)
    n_channels, N = signal.shape

    target_depth = 1.5  # meters
    if method == "AMF":
        basis_functions, t_Tau = __anderson_basis_functions(
            filter_width, dov=target_depth / drone_velocity
        )
    elif method == "CMF":
        basis_functions, t_Tau = __cable_basis_functions(
            filter_width, dov=target_depth / drone_velocity
        )
    elif method == "GradMF":
        basis_functions, t_Tau = __gradient_basis_functions(
            filter_width, dov=target_depth / drone_velocity
        )
    else:
        assert False, f"Unknown method '{method}'. Valid options: 'AMF', 'CMF', 'GradMF'."
    number_basis = len(basis_functions)
    basis_functions = basis_functions[:, ::-1]

    # --- interpolate each basis onto the FULL signal time grid, centered
    t_center = time[N // 2]  # midpoint of the signal time axis
    t_Tau_centered = t_Tau + t_center  # shift basis to be centered on signal

    basis_expanded = np.zeros((number_basis, N))
    for i in range(number_basis):
        basis_expanded[i] = np.interp(
            time, t_Tau_centered, basis_functions[i], left=0, right=0
        )

    # --- full convolution for all channels and all bases
    conv_full = np.zeros((number_basis, n_channels, N))
    for i in range(number_basis):
        for j in range(n_channels):
            conv_full[i, j] = __Matched_filter(basis_expanded[i], signal[j])
            import matplotlib.pyplot as plt

            plt.figure()
            plt.plot(signal[j])
            plt.plot(basis_expanded[i])
            plt.show(block=False)

    if signal.shape[0] == 1:
        conv_full = conv_full[:, 0, :]

    return basis_expanded, conv_full


# endregion
