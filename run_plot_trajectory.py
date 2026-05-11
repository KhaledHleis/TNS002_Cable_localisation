import argparse

from helper.trajectory_reader import read_trajectory
from helper.plot_functions import PlotFunctions as pf

pf.PAPER_MODE = False
FS = 50.0  # Sampling frequency (Hz)
DT = 1.0 / FS  # Time step (s)

"""    return Trajectory(
    timestamp    = df["timestamp"],
    longitude    = df["longitude"],
    latitude     = df["latitude"],
    velocity     = np.stack((df["ve"], df["vn"], df["vd"]), axis=0),
    heading      = df["heading"],
    magnetic     = np.stack((df["mag_x"], df["mag_y"], df["mag_z"]), axis=0),
    mag_norm     = df["mag"],
    acceleration = np.stack((df["acc_x"], df["acc_y"], df["acc_z"]), axis=0),
)
"""


def main(args):
    traj = read_trajectory(args.file, DT)
    pf.plot_trajectory(traj.longitude, traj.latitude, traj.mag_norm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot trajectory data.")
    parser.add_argument("file", help="Path to the trajectory CSV file.")
    args = parser.parse_args()
    main(args)
    input("Press Enter to exit...")
