import numpy as np
from scipy.signal import butter, filtfilt
from scipy.optimize import minimize
import rasterio


# region #* filtering
def low_pass(signal, cutoff, fs, order=4):
    """
    Apply a low-pass Butterworth filter.

    Parameters
    ----------
    signal : array-like
        Input signal
    cutoff : float
        Cutoff frequency (Hz)
    fs : float
        Sampling frequency (Hz)
    order : int, optional
        Filter order (default=4)

    Returns
    -------
    filtered_signal : ndarray
        Low-pass filtered signal
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist

    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, signal)


# endregion


# region #* calibration by linear relationship
def linear_relation_parameters(signal_1: np.ndarray, signal_2: np.ndarray):
    """
    Estimate linear relation parameters such that:
        signal_2 ≈ a * signal_1 + b

    Args:
        signal_1 (np.ndarray): input signal
        signal_2 (np.ndarray): desired output signal

    Returns:
        a (float), b (float): model parameters
    """
    if not isinstance(signal_1, np.ndarray) or not isinstance(signal_2, np.ndarray):
        raise TypeError("signal_1 and signal_2 must be numpy.ndarray")

    if signal_1.shape != signal_2.shape:
        raise ValueError("signal_1 and signal_2 must have the same shape")

    x = signal_1
    y = signal_2

    def cost(params):
        a, b = params
        return np.mean((y - (a * x + b)) ** 2)

    initial_guess = np.array([1.0, 0.0])

    result = minimize(cost, initial_guess, method="BFGS")

    print("optimization results:", result.x)
    a, b = result.x
    return float(a), float(b)


def apply_linear_relation(input_function: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Apply linear relation to input signal.

    Args:
        input_function (np.ndarray): input signal
        a (float): slope
        b (float): offset

    Returns:
        np.ndarray: corrected input
    """
    if not isinstance(input_function, np.ndarray):
        raise TypeError("input_function must be a numpy.ndarray")

    corrected_input = a * input_function + b
    return corrected_input


# endregion

# region #* detection


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


def __anderson_basis_functions(N, dov=1):
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


def __cable_basis_functions(N, dov=1, theta=90):
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


def Normalize(sig):
    signal = sig.astype(float)  # ensure float for safe division
    signal -= np.mean(signal, axis=-1, keepdims=True)
    signal /= np.max(np.abs(signal), axis=-1, keepdims=True)
    return signal


def Normalize_point(sig, index):
    signal = sig.astype(float)  # ensure float for safe division
    signal /= sig[index]
    signal -= 1
    return signal


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
    else:
        return -1
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

            # plt.figure()
            # plt.plot(signal[j])
            # plt.plot(basis_expanded[i])
            # plt.show(block=False)

    if signal.shape[0] == 1:
        conv_full = conv_full[:, 0, :]

    return basis_expanded, conv_full


def detect_anomalies_windowed(
    signal: np.ndarray,
    velocity: np.ndarray,  # (3, N)
    number_basis: int = 3,
    filter_width: int = 80,
    hop_size: int | None = None,
    method: str = "AMF",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Overlapping-chunk matched filtering with velocity-aware Anderson basis.

    Parameters
    ----------
    signal : np.ndarray
        Shape (n_channels, N) or (N,)
    time : np.ndarray
        Shape (N,)
    velocity : np.ndarray
        Shape (3, N) velocity vector (used instead of drone_velocity)
    number_basis : int
        Number of Anderson basis functions
    filter_width : int
        Chunk size (also basis length)
    hop_size : int
        Step between chunks (default = filter_width // 2)
    method : str
        Detection method

    Returns
    -------
    basis_expanded : np.ndarray
        Shape (number_basis, filter_width)
    decision_signal : np.ndarray
        Shape (number_basis, n_channels, N)
        Per-sample anomaly decision (same length as input)
    """

    signal = np.atleast_2d(signal)
    n_channels, N = signal.shape

    if hop_size is None:
        hop_size = filter_width // 2

    assert velocity.shape == (3, N), "velocity must have shape (3, N)"

    target_depth = 1.5  # meters

    # Output accumulator
    decision_signal = np.zeros((number_basis, n_channels, N))
    overlap_count = np.zeros(N)

    if method != "AMF":
        return -1

    # --- sliding window processing
    for start in range(0, N - filter_width + 1, hop_size):
        stop = start + filter_width

        signal_chunk = signal[:, start:stop]
        velocity_chunk = velocity[:, start:stop]

        # --- velocity magnitude (used instead of estimation)
        v_mag = np.linalg.norm(velocity_chunk, axis=0)
        mean_velocity = np.mean(v_mag) + 1e-8  # avoid divide-by-zero

        # --- Anderson basis for this chunk
        basis_functions, _ = __anderson_basis_functions(
            filter_width, number_basis, dov=target_depth / mean_velocity
        )

        basis_functions = basis_functions[:, ::-1]  # matched filter
        basis_expanded = basis_functions.copy()

        # --- matched filtering per channel
        for i in range(number_basis):
            for j in range(n_channels):
                mf_out = __Matched_filter(basis_expanded[i], signal_chunk[j])

                # decision_signal[i, j, start:stop] += np.ones_like(mf_out) * np.max(abs(mf_out))
                decision_signal[i, j, start:stop] += mf_out

        overlap_count[start:stop] += 1

    # --- normalize overlapping contributions
    overlap_count[overlap_count == 0] = 1
    decision_signal /= overlap_count

    # --- single-channel compatibility
    if n_channels == 1:
        decision_signal = decision_signal[:, 0, :]

    return basis_expanded, decision_signal


# endregion

# region #* aiding functions for geographic and geomagnetic data
from pyproj import Transformer, Proj

def latlon_to_cartesian(
    longitude: np.ndarray,
    latitude:  np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert WGS-84 longitude/latitude to a local Cartesian frame (m).
    Origin = first trajectory point.  East → +x, North → +y.
    """
    lon0, lat0 = float(longitude[0]), float(latitude[0])
    proj = Proj(proj="aeqd", lat_0=lat0, lon_0=lon0, datum="WGS84", units="m")
    x_cart, y_cart = proj(longitude, latitude)
    return np.asarray(x_cart, dtype=float), np.asarray(y_cart, dtype=float)



def ned_to_frd(v_n, v_e, v_d, heading_rad):
    v_f = np.cos(heading_rad) * v_n + np.sin(heading_rad) * v_e
    v_r = -np.sin(heading_rad) * v_n + np.cos(heading_rad) * v_e
    return v_f, v_r, v_d


def get_depth_column(depth_tif, longitude, latitude):
    with rasterio.open(depth_tif) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)

        x, y = transformer.transform(longitude, latitude)
        points = np.column_stack((x, y))

        depth = np.array([v[0] for v in src.sample(points)])

    return depth

def plot_bouees(magnetic_coords=None):
    import matplotlib.pyplot as plt

    if magnetic_coords is None:
        magnetic_coords = [
            (48.492313209033945, -4.503827049110521),#a
            (48.4922709,-4.5045318),#b
            # (48.4923421,-4.50450887),#c
            # (48.4925255,-4.5038868),#d
            (48.4923192,-4.5045399),#c
            (48.4925066,-4.5038826),#d
        ]

    mg_lon = [lon for lat, lon in magnetic_coords]
    mg_lat = [lat for lat, lon in magnetic_coords]
    
    # Magnetic cable buoys: diamond marker
    plt.scatter(
        mg_lon,
        mg_lat,
        marker="D",
        color="tab:blue",
        s=100,
        label="approximated cable position",
    )

    # Connect all magnetic buoys to form a loop (a -> b -> c -> d -> a)
    if len(mg_lon) >= 2:
        loop_lon = mg_lon + [mg_lon[0]]  # Add first point at end to close the loop
        loop_lat = mg_lat + [mg_lat[0]]
        plt.plot(
            loop_lon, loop_lat, color="tab:blue", linestyle="-", linewidth=1
        )
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Returns distance between two GPS points in meters.
    """
    import math
    R = 6371000  # Earth radius in meters

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (math.sin(dphi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) *
         math.sin(dlambda / 2) ** 2)

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

def plot_bouees_jan(acoustic_coords=None, ponton_coord=None, magnetic_coords=None):
    import matplotlib.pyplot as plt

    # Default coordinates (lat, lon) as provided
    if acoustic_coords is None:
        acoustic_coords = [
            (48.49209220724089, -4.503754158369759),
            (48.49210461149053, -4.504148105330711),
        ]
    if ponton_coord is None:
        ponton_coord = (48.492313209033945, -4.503827049110521)
    if magnetic_coords is None:
        magnetic_coords = [
            (48.49227909392259, -4.50434496199812),
            (48.492394695036225, -4.504234054879934),
        ]

    # Convert (lat, lon) -> plot as (x=lon, y=lat)
    ac_lon = [lon for lat, lon in acoustic_coords]
    ac_lat = [lat for lat, lon in acoustic_coords]

    pt_lon = ponton_coord[1]
    pt_lat = ponton_coord[0]

    mg_lon = [lon for lat, lon in magnetic_coords]
    mg_lat = [lat for lat, lon in magnetic_coords]

    # Acoustic cable buoys: triangle marker
    plt.scatter(
        ac_lon,
        ac_lat,
        marker="^",
        color="tab:red",
        s=100,
        label="Bouées câble acoustique",
    )
    # Connect acoustic buoys together
    if len(ac_lon) >= 2:
        plt.plot(ac_lon, ac_lat, color="tab:red", linestyle="-", linewidth=1)

    # Ponton: x marker
    plt.scatter(
        [pt_lon], [pt_lat], marker="x", color="tab:green", s=100, label="Ponton"
    )

    # Magnetic cable buoys: diamond marker
    plt.scatter(
        mg_lon,
        mg_lat,
        marker="D",
        color="tab:blue",
        s=100,
        label="Bouées câble magnétique",
    )

    # Connect ponton and magnetic buoys to form a triangle
    if len(mg_lon) >= 2:
        triangle_lon = [pt_lon, mg_lon[0], mg_lon[1], pt_lon]
        triangle_lat = [pt_lat, mg_lat[0], mg_lat[1], pt_lat]
        plt.plot(
            triangle_lon, triangle_lat, color="tab:blue", linestyle="-", linewidth=1
        )


from pyIGRF import igrf_value


def get_magfield_ref_components(longitude, latitude):
    """
    Magnetic field reference from IGRF (ground robot).

    Args:
        longitude (float or np.ndarray): degrees (EPSG:4326)
        latitude (float or np.ndarray): degrees (EPSG:4326)

    Returns:
        np.ndarray:
            - shape (4,) for single point
            - shape (N, 4) for trajectory
            Components: [X, Y, Z, F] in µT
    """

    scalar_input = np.isscalar(longitude) and np.isscalar(latitude)

    lon = np.atleast_1d(longitude).astype(float)
    lat = np.atleast_1d(latitude).astype(float)

    if lon.shape != lat.shape:
        raise ValueError("longitude and latitude must have same shape")

    alt_km = np.zeros_like(lon)  # ground level
    year = 2025.0

    B = np.empty((lon.size, 3))

    for i in range(lon.size):
        _, _, _, X, Y, Z, F = igrf_value(lat[i], lon[i], alt_km[i], year)
        B[i] = [X, Y, Z]

    B_uT = B * 1e-3  # nT → µT

    return B_uT[0] if scalar_input else B_uT


# endregion


# region #* peak-picking
def local_maxima_indices_plateau(signal: np.ndarray) -> np.ndarray:
    """
    Return indices of local maxima, including flat plateaus.

    Peaks are kept only if:
        peak_value >= threshold

    Args:
        signal (np.ndarray): 1D input signal
        threshold (float): Minimum peak amplitude to keep

    Returns:
        np.ndarray: indices of significant local maxima
    """

    signal = np.asarray(signal)

    if signal.ndim != 1 or signal.size < 3:
        return np.array([], dtype=int)

    maxima = []
    i = 1

    while i < signal.size - 1:
        if signal[i] > signal[i - 1]:
            start = i
            while i < signal.size - 2 and signal[i] == signal[i + 1]:
                i += 1
            end = i

            if signal[end] > signal[end + 1]:
                idx = (start + end) // 2
                maxima.append(idx)

        i += 1

    return np.array(maxima, dtype=int)


def local_maxima_indices_diff(signal: np.ndarray) -> np.ndarray:
    """
    Simple local maxima detector using first differences.
    """
    signal = np.asarray(signal)

    if signal.ndim != 1 or signal.size < 3:
        return np.array([], dtype=int)

    diff = np.diff(signal)
    maxima = np.where((diff[:-1] > 0) & (diff[1:] < 0))[0] + 1

    return maxima.astype(int)


def local_maxima_indices_robust(
    signal: np.ndarray,
    min_prominence: float = 0.0,
    min_width: int = 1,
    smooth_window: int = 1,
) -> np.ndarray:
    """
    Robust local maxima detector resistant to noise.

    A peak is kept if:
      - peak_value >= threshold
      - peak_value - local_baseline >= min_prominence
      - plateau width >= min_width

    Args:
        signal (np.ndarray): 1D input signal
        threshold (float): Minimum absolute peak amplitude
        min_prominence (float): Minimum peak prominence
        min_width (int): Minimum plateau width (samples)
        smooth_window (int): Moving average window (odd, >=1)

    Returns:
        np.ndarray: indices of significant local maxima
    """

    signal = np.asarray(signal, dtype=float)

    if signal.ndim != 1 or signal.size < 3:
        return np.array([], dtype=int)

    # --- Optional smoothing ---
    if smooth_window > 1:
        w = smooth_window
        kernel = np.ones(w) / w
        signal = np.convolve(signal, kernel, mode="same")

    maxima = []
    i = 1
    n = signal.size

    while i < n - 1:
        if signal[i] > signal[i - 1]:
            start = i
            while i < n - 1 and signal[i] == signal[i + 1]:
                i += 1
            end = i

            if end < n - 1 and signal[end] > signal[end + 1]:
                width = end - start + 1
                idx = (start + end) // 2
                peak_val = signal[idx]

                # Local baseline (left & right minima)
                left_min = signal[max(0, start - min_width) : start].min(
                    initial=peak_val
                )
                right_min = signal[end + 1 : min(n, end + 1 + min_width)].min(
                    initial=peak_val
                )
                baseline = max(left_min, right_min)

                prominence = peak_val - baseline

                if prominence >= min_prominence and width >= min_width:
                    maxima.append(idx)

        i += 1

    return np.array(maxima, dtype=int)


import numpy as np


def detect_anomalies_with_confidence(
    rx_merged,
    conf=None,
    threshold_peaks=0.5,
    threshold_holes=0.5,
    conf_max=0.5,
):
    """
    Detect anomalies (peaks and holes) using confidence filtering
    and per-class [0,1] amplitude normalization.

    Steps:
    1) Detect peaks and holes
    2) Filter by confidence
    3) Normalize magnitudes to [0,1] separately
    4) Apply thresholds

    Parameters
    ----------
    rx_merged : np.ndarray
        Signal used for anomaly detection.
    conf : np.ndarray
        Confidence values in [0,1], same length as rx_merged.
    threshold_peaks : float
        Threshold on normalized peak magnitude [0,1].
    threshold_holes : float
        Threshold on normalized hole magnitude [0,1].
    conf_max : float
        Maximum confidence allowed for anomaly candidates.

    Returns
    -------
    anomaly_indexes : np.ndarray
        Final anomaly indices.
    peaks_final : np.ndarray
        Selected peak indices.
    holes_final : np.ndarray
        Selected hole indices.
    """

    # --- Detect candidates ---
    peaks_idx_all = local_maxima_indices_robust(rx_merged)
    holes_idx_all = local_maxima_indices_robust(-rx_merged)

    if conf is not None:
        # --- Confidence filtering ---
        peaks_idx = peaks_idx_all[conf[peaks_idx_all] < conf_max]
        holes_idx = holes_idx_all[conf[holes_idx_all] < conf_max]
    else:
        peaks_idx = peaks_idx_all
        holes_idx = holes_idx_all

    # --- Extract amplitudes ---
    rx_peaks = rx_merged[peaks_idx]
    rx_holes = -rx_merged[holes_idx]

    # --- Normalize to [0,1] separately ---
    if len(rx_peaks) > 0:
        peaks_norm = rx_peaks - np.min(rx_peaks)
        peaks_norm /= np.max(peaks_norm)
    else:
        peaks_norm = np.array([])

    if len(rx_holes) > 0:
        holes_norm = rx_holes - np.min(rx_holes)
        holes_norm /= np.max(holes_norm)
    else:
        holes_norm = np.array([])

    # --- Thresholding ---
    peaks_final = peaks_idx[peaks_norm >= threshold_peaks]
    holes_final = holes_idx[holes_norm >= threshold_holes]

    # --- Combine ---
    anomaly_indexes = np.concatenate([peaks_final, holes_final])

    return anomaly_indexes, peaks_final, holes_final


def overhang(rectangular_signal: np.ndarray, overhang_size: int) -> np.ndarray:
    """
    Fill the gaps in a rectangular signal of size smaller or equal to overhang_size.

    Parameters
    ----------
    rectangular_signal : np.ndarray
        1D input signal
    overhang_size : int
        Maximum number of samples to fill

    Returns
    -------
    np.ndarray
        Signal with overhang added
    """
    if rectangular_signal.ndim != 1:
        raise ValueError("rectangular_signal must be 1D")

    if overhang_size <= 0:
        return rectangular_signal.copy()

    signal = rectangular_signal.copy()

    # Convert to binary mask (non-zero = 1)
    mask = signal != 0

    # Find transitions
    diff = np.diff(mask.astype(int))

    # Start/end indices of zero gaps
    gap_starts = np.where(diff == -1)[0] + 1
    gap_ends = np.where(diff == 1)[0] + 1

    # Ensure gaps are internal (ignore leading/trailing zeros)
    if mask[0] == 0 and gap_ends.size > 0:
        gap_ends = gap_ends[1:]
    if mask[-1] == 0 and gap_starts.size > 0:
        gap_starts = gap_starts[:-1]

    # Fill small gaps
    for start, end in zip(gap_starts, gap_ends):
        gap_length = end - start
        if gap_length <= overhang_size:
            signal[start:end] = np.max(signal)

    return signal


# endregion

# region # HACK: section of functions used for debugging

def matched_filter_anderson(
    signal: np.ndarray,
    time: np.ndarray,
    basis_index: int | None = 1,
    filter_width: int = 80,
    lag: int = 0,
    drone_velocity: float = 1.0,
    method: str = "CMF",
) -> tuple[np.ndarray, float | np.ndarray]:
    """
    Time-correct matched filter using Anderson/Cable basis functions.

    Parameters
    ----------
    signal : np.ndarray
        Shape (1, N) or (N,) signal values.
    time : np.ndarray
        Shape (N,) time axis for the signal.
    basis_index : int or None
        If int  → use only that single basis (0-indexed).
                  Returns (basis_expanded shape (1,N), conv_value float)
                  — identical to the original debug output.
        If None → run all bases and return their matched-filter norm
                  along the basis axis.
                  Returns (basis_expanded shape (n_basis,N), norms shape (n_basis,))
    filter_width : int
        Number of basis samples.
    lag : int
        Lag in *signal samples*. Shifts the basis center on the time axis.
    drone_velocity : float
        Used to compute dov = target_depth / drone_velocity.
    method : str
        "AMF" for Anderson basis, "CMF" for cable basis.

    Returns
    -------
    basis_out : np.ndarray
        basis_index is int  → shape (1, N)
        basis_index is None → shape (n_basis, N)
    conv_out : float | np.ndarray
        basis_index is int  → float  (same as original)
        basis_index is None → shape (n_basis,) norms along basis axis
    """
    signal = np.atleast_2d(signal)
    N = signal.shape[-1]
    target_depth = 1.5  # metres

    # --- build basis functions ----------------------------------------------
    dov = target_depth / drone_velocity
    if method == "AMF":
        basis_functions, t_Tau = __anderson_basis_functions(filter_width, dov=dov)
    elif method == "CMF":
        basis_functions, t_Tau = __cable_basis_functions(filter_width, dov=dov)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'AMF' or 'CMF'.")

    # reverse all bases for matched filtering
    basis_functions = basis_functions[:, ::-1]
    number_basis = len(basis_functions)

    # --- apply lag ----------------------------------------------------------
    dt = time[1] - time[0]
    t_Tau_shifted = t_Tau + lag * dt

    # --- select which bases to process --------------------------------------
    indices = list(range(number_basis)) if basis_index is None else [basis_index]

    # --- interpolate onto signal time grid ----------------------------------
    basis_expanded = np.zeros((len(indices), N))
    for out_i, src_i in enumerate(indices):
        basis_expanded[out_i] = np.interp(
            time, t_Tau_shifted, basis_functions[src_i], left=0.0, right=0.0
        )

    # --- matched filter output ----------------------------------------------
    if basis_index is not None:
        # ✅ identical output shape to original: (1, N), float
        conv_value: float = float(np.sum(np.linalg.norm(signal * basis_expanded)))
        return basis_expanded[-1], conv_value
    else:
        # norm of each basis's filter response over channels
        # per_channel: (n_basis, n_channels), norm over channels → (n_basis,)
        conv = 0
        for basis in basis_expanded:
            conv += (basis * signal)**2 # (n_basis, n_channels)
        return basis_expanded, conv

# endregion
