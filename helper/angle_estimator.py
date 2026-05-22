import numpy as np

def batch_estimate_angles(detection_indices_1: np.ndarray, detection_indices_2: np.ndarray, Fs, drone_velocity: float = 1, l: float = 1) -> np.ndarray:
    """Batch estimates angles of incidence from two arrays of detection indices.

    Args:
        detection_indices_1 (np.ndarray): array of indices for the first sensor
        detection_indices_2 (np.ndarray): array of indices for the second sensor
        Fs (int): sampling frequency in Hz
        drone_velocity (float): velocity of the drone in m/s
        l (float): distance between the two sensors in meters
    Returns:
        np.ndarray: array of estimated angles of incidence in degrees
    """
    time_diffs = np.abs(detection_indices_2 - detection_indices_1) / Fs
    angles = np.arctan(l/(time_diffs * drone_velocity)) * (180 / np.pi)  # Convert radians to degrees
    return angles

def estimate_angle(detection_index_1: int, detection_index_2: int, Fs, drone_velocity: float = 1, l: float = 1) -> float:
    """Estimates angle of incidence from two detection indices.

    Args:
        detection_index_1 (int): index of the first detection (e.g., from sensor 1)
        detection_index_2 (int): index of the second detection (e.g., from sensor 2)
        Fs (int): sampling frequency in Hz
        drone_velocity (float): velocity of the drone in m/s
        l (float): distance between the two sensors in meters

    Returns:
        float: estimated angle of incidence in degrees
    """
    time_diff = abs(detection_index_2 - detection_index_1) / Fs
    # Assuming a simple model where time difference corresponds to angle
    # For example, if the object moves at a constant speed and the sensors are spaced apart,
    # we can use the time difference to estimate the angle of incidence.
    # This is a placeholder formula and should be replaced with the actual model based on sensor geometry.
    angle = np.arctan(l/(time_diff * drone_velocity)) * (180 / np.pi)  # Convert radians to degrees
    return angle
