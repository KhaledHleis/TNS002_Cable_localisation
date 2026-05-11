from helper.utilities import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import functools
import contextlib


# ---------------------------------------------------------------------------
# Paper-mode style context
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _paper_style_context():
    """
    Context manager that temporarily applies publication-quality matplotlib
    settings (IEEE / Elsevier style) and restores the original rcParams on exit.
    """
    paper_rc = {
        # --- Figure size & DPI ---
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "figure.figsize": (6.4, 4.0),  # ~one-column IEEE width in inches
        # --- Font ---
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        # --- Lines & markers ---
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        # --- Axes & spines ---
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.4,
        "grid.alpha": 0.6,
        # --- Ticks ---
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        # --- Legend ---
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        # NOTE: constrained_layout is intentionally omitted here.
        # It conflicts with tight_layout() and colorbars (raises RuntimeError).
        # tight_layout() called inside each method handles spacing instead.
        "figure.constrained_layout.use": False,
    }

    original = {k: mpl.rcParams[k] for k in paper_rc if k in mpl.rcParams}
    try:
        mpl.rcParams.update(paper_rc)
        yield
    finally:
        mpl.rcParams.update(original)


# ---------------------------------------------------------------------------
# Decorator factory
# ---------------------------------------------------------------------------


def paper_style(method):
    """
    Decorator that wraps a PlotFunctions static method so that when
    ``PlotFunctions.PAPER_MODE is True`` the plot is rendered with
    publication-quality settings; otherwise it renders with normal defaults.

    Usage
    -----
    Decorate any static method that calls matplotlib:

        @staticmethod
        @paper_style
        def plot_something(...):
            ...

    Toggle globally with:

        PlotFunctions.PAPER_MODE = True   # publication style
        PlotFunctions.PAPER_MODE = False  # normal/programming style
    """

    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        if PlotFunctions.PAPER_MODE:
            with _paper_style_context():
                return method(*args, **kwargs)
        else:
            return method(*args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class PlotFunctions:
    """
    Collection of static plot helpers for magnetic-field / drone-nav analysis.

    Class-level switch
    ------------------
    PAPER_MODE : bool
        Set to ``True`` to render all plots in publication-quality style
        (serif fonts, tight layout, higher DPI, clean spines).
        Set to ``False`` (default) for normal interactive / programming plots.

    Example
    -------
    >>> PlotFunctions.PAPER_MODE = True
    >>> PlotFunctions.plot_calibration(time, before, after, truth)
    >>> PlotFunctions.PAPER_MODE = False   # back to normal
    """

    PAPER_MODE: bool = False  # ← flip this to True for paper/article figures

    @staticmethod
    @paper_style
    def plot_correction_cap(
        time, sig_mag, sig_calib, sig_heading_abs, slop=None, bias=None, block=False
    ):
        """
        Plot magnetic field correction and GNSS heading before and after calibration.

        Parameters
        ----------
        time : array-like
            Time values corresponding to the measurements.
        sig_mag : array-like
            Magnetic field magnitude measurements before calibration.
        sig_calib : array-like
            Estimated correction term or magnetic field magnitude after calibration.
        sig_heading_abs : array-like
            GNSS heading values (absolute) corresponding to the measurements.
        slop : float, optional
            Slope value from calibration (if available). Default is None.
        bias : float, optional
            Bias value from calibration (if available). Default is None.
        block : bool, optional
            If True, block execution until the plot window is closed. Default is False.

        Returns
        -------
        None
        """
        i = 2
        plt.figure()

        if slop is not None and bias is not None:
            i = 3
            plt.subplot(i, 1, 3)
            plt.scatter(sig_mag, sig_heading_abs, label="Before Calibration", alpha=0.5)
            plt.scatter(
                sig_calib, sig_heading_abs, label="After Calibration", alpha=0.5
            )
            plt.legend()
            plt.grid()
            plt.xlabel("Magnetic Field Magnitude (nT)")
            plt.ylabel("GNSS Heading")
            plt.title(f"Calibration Scatter Plot (slope={slop:.4f}, bias={bias:.4f})")

        plt.subplot(i, 1, 1)
        plt.plot(time, sig_mag, label="Magnetic Field Magnitude")
        plt.plot(time, sig_calib, label="Estimated Correction Term")
        plt.title("Magnetic Field Magnitude recorded vs estimated from cap")
        plt.xlabel("Time (s)")
        plt.ylabel("Magnetic Field (nT)")
        plt.legend()
        plt.grid()

        plt.subplot(i, 1, 2)
        plt.plot(time, sig_heading_abs, label="GNSS Heading absolute")
        plt.title("GNSS Heading vs Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Heading (degrees)")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show(block=block)

    @staticmethod
    @paper_style
    def plot_calibration(
        time,
        mag_before,
        mag_after,
        mag_truth,
        cost_curve=None,
        components=None,
        block=False,
        caption="",
    ):
        """
        Plot magnetic field calibration results.

        Parameters
        ----------
        time : array-like
            Time values corresponding to the measurements.
        mag_before : array-like
            Magnetic field magnitude measurements before calibration.
        mag_after : array-like
            Magnetic field magnitude measurements after calibration.
        mag_truth : array-like
            True magnetic field magnitude values (ground truth).
        cost_curve : array-like, optional
            Calibration cost curve values (if available). Default is None.
        components : list of array-like, optional
            Individual components of the magnetic field (if available). Default is None.
        block : bool, optional
            If True, block execution until the plot window is closed. Default is False.
        caption : str, optional
            Caption to display at the bottom of the plot. Default is an empty string.

        Returns
        -------
        None
        """
        plt.figure()
        if cost_curve is not None:
            plt.subplot(2, 1, 2)
            plt.plot(cost_curve)
            plt.title("Calibration Cost Curve")
            plt.xlabel("Iteration")
            plt.yscale("log")
            plt.ylabel("Cost")
            plt.grid()
            plt.subplot(2, 1, 1)
        plt.plot(time, mag_before, label="Before Calibration")
        plt.plot(time[: len(mag_after)], mag_after, label="After Calibration")
        plt.plot(time, mag_truth, label="True Values", linestyle="--")
        if components is not None:
            plt.plot(
                time[: len(mag_after)],
                components[0],
                label="Calibrated X",
                linestyle=":",
                color="red",
            )
            plt.plot(
                time[: len(mag_after)],
                components[1],
                label="Calibrated Y",
                linestyle=":",
                color="blue",
            )
            plt.plot(
                time[: len(mag_after)],
                components[2],
                label="Calibrated Z",
                linestyle=":",
                color="green",
            )

        if caption:
            plt.subplots_adjust(bottom=0.18)
            plt.gcf().text(0.5, 0.04, caption, ha="center", fontsize=9)

        plt.title("Magnetic Field Calibration")
        plt.xlabel("Time (s)")
        plt.ylabel("Magnetic Field (nT)")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show(block=block)

    @staticmethod
    @paper_style
    def plot_calibration_effect(
        time,
        magvec_before,
        magvec_after,
        mag_ref=None,
        components=None,
        block=False,
        caption="",
    ):
        """
        Plot the effect of calibration on the magnetic field vector and its norm.

        Parameters
        ----------
        time : array-like
            Time values corresponding to the measurements.
        magvec_before : array-like, shape (3, N)
            Magnetic field vector before calibration (X, Y, Z) over time.
        magvec_after : array-like, shape (3, N)
            Magnetic field vector after calibration (X, Y, Z) over time.
        mag_ref : array-like, optional
            Reference/true magnetic field magnitude over time (length N). If provided,
            it will be plotted on the norm subplot.
        components : list or tuple of array-like, optional
            Optional additional component series to plot for each axis.
        block : bool, optional
            If True, block execution until the plot window is closed. Default is False.
        caption : str, optional
            Caption text to display at the bottom of the figure. Default is "".

        Returns
        -------
        None
        """
        fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
        ax_norm, ax_x, ax_y, ax_z = axes

        magvec_before = np.asarray(magvec_before)
        magvec_after = np.asarray(magvec_after)

        # --- Norm subplot ---
        mag_norm_before = np.linalg.norm(magvec_before, axis=0)
        mag_norm_after = np.linalg.norm(magvec_after, axis=0)
        ax_norm.plot(time, mag_norm_before, label="Before Calibration")
        ax_norm.plot(time, mag_norm_after, label="After Calibration")
        if mag_ref is not None:
            ax_norm.plot(
                time[: len(mag_ref)],
                mag_ref,
                label="Reference Magnetic Field",
                linestyle="--",
            )
        ax_norm.set_title("Magnetic Field Norm - Before vs After Calibration")
        ax_norm.set_ylabel("Magnetic Field (nT)")
        ax_norm.legend()
        ax_norm.grid()

        # --- X component ---
        ax_x.plot(time, magvec_before[0, :], label="Before Calibration X")
        ax_x.plot(time, magvec_after[0, :], label="After Calibration X")
        if (
            components is not None
            and len(components) >= 1
            and components[0] is not None
        ):
            ax_x.plot(
                time[: len(components[0])],
                components[0],
                label="Calibrated X",
                linestyle=":",
                color="red",
            )
        ax_x.set_title("X Component")
        ax_x.set_ylabel("Field X (nT)")
        ax_x.legend()
        ax_x.grid()

        # --- Y component ---
        ax_y.plot(time, magvec_before[1, :], label="Before Calibration Y")
        ax_y.plot(time, magvec_after[1, :], label="After Calibration Y")
        if (
            components is not None
            and len(components) >= 2
            and components[1] is not None
        ):
            ax_y.plot(
                time[: len(components[1])],
                components[1],
                label="Calibrated Y",
                linestyle=":",
                color="blue",
            )
        ax_y.set_title("Y Component")
        ax_y.set_ylabel("Field Y (nT)")
        ax_y.legend()
        ax_y.grid()

        # --- Z component ---
        ax_z.plot(time, magvec_before[2, :], label="Before Calibration Z")
        ax_z.plot(time, magvec_after[2, :], label="After Calibration Z")
        if (
            components is not None
            and len(components) >= 3
            and components[2] is not None
        ):
            ax_z.plot(
                time[: len(components[2])],
                components[2],
                label="Calibrated Z",
                linestyle=":",
                color="green",
            )
        ax_z.set_title("Z Component")
        ax_z.set_ylabel("Field Z (nT)")
        ax_z.legend()
        ax_z.grid()

        ax_z.set_xlabel("Time (s)")

        if caption:
            plt.subplots_adjust(bottom=0.18)
            fig.text(0.5, 0.04, caption, ha="center", fontsize=9)

        plt.tight_layout()
        plt.show(block=block)

    @staticmethod
    @paper_style
    def plot_trajectory(
        longitude,
        latitude,
        sig_mag,
        anomaly_indexes=np.array([]),
        block=False,
        caption="",
    ):
        """
        Plot trajectory with magnetic field magnitude and anomalies.

        Parameters
        ----------
        longitude : array-like
            Array of longitude values for each measurement point.
        latitude : array-like
            Array of latitude values for each measurement point.
        sig_mag : array-like
            Magnetic field magnitude measurements at each point.
        anomaly_indexes : array-like, optional
            Indexes of points considered anomalies (default: empty array).
        block : bool, optional
            If True, block execution until the plot window is closed. Default is False.
        caption : str, optional
            Caption text to display at the bottom of the figure. Default is "".

        Returns
        -------
        None
        """
        plt.figure()
        sc = plt.scatter(
            longitude, latitude, c=sig_mag, cmap="viridis", s=20, alpha=0.8
        )
        plt.scatter(longitude[0], latitude[0], marker="o", c="green", s=50, label="Start")
        plt.scatter(longitude[-1], latitude[-1], marker="X", c="blue", s=50, label="End")
        if anomaly_indexes.size > 0:
            plt.scatter(
                longitude[anomaly_indexes],
                latitude[anomaly_indexes],
                marker="x",
                c="red",
                s=20,
                linewidths=2,
                label="Anomalies",
            )

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Magnetic Anomaly Magnitude along the USV trajectory")
        plt.axis("equal")
        plt.grid(True)
        plot_bouees()
        cbar = plt.colorbar(sc)
        cbar.set_label("Magnetic Anomaly magnitude in µT")

        if caption:
            plt.subplots_adjust(bottom=0.18)
            plt.gcf().text(0.5, 0.04, caption, ha="center", fontsize=9)

        plt.legend()
        plt.tight_layout()
        plt.show(block=block)

    @staticmethod
    @paper_style
    def plot_trajectory_with_metrics(
        longitude,
        latitude,
        sig_mag,
        cables_lon,
        cables_lat,
        anomaly_indexes=np.array([]),
        match_window=30,
        block=False,
        caption="",
    ):
        """
        Plot trajectory, anomalies, cable crossings, matching results,
        and anomaly detection metrics.

        PARAMETERS
        ----------
        longitude : array-like
        latitude : array-like
        sig_mag : array-like
        cables_lon : array-like
            Cable endpoints longitude (pairs start/end)
        cables_lat : array-like
            Cable endpoints latitude (pairs start/end)
        anomaly_indexes : array-like
            Detected anomaly indexes
        match_window : int
            Max trajectory-index distance for valid match
        block : bool
        caption : str

        RETURNS
        -------
        dict containing metrics
        """

        import numpy as np
        import matplotlib.pyplot as plt

        # =====================================================
        # Build cable segments
        # =====================================================
        assert len(cables_lon) % 2 == 0, "Cable points must come in pairs"

        cable_segments = [
            (
                cables_lon[i],
                cables_lat[i],
                cables_lon[i + 1],
                cables_lat[i + 1],
            )
            for i in range(0, len(cables_lon), 2)
        ]

        # =====================================================
        # Geometry helpers
        # =====================================================
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        def segments_intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

        # =====================================================
        # Find true cable crossings
        # =====================================================
        true_crossings = []

        for i in range(len(longitude) - 1):

            A = (longitude[i], latitude[i])
            B = (longitude[i + 1], latitude[i + 1])

            for lon1, lat1, lon2, lat2 in cable_segments:
                C = (lon1, lat1)
                D = (lon2, lat2)

                if segments_intersect(A, B, C, D):
                    true_crossings.append(i + 0.5)
                    break

        true_crossings = sorted(true_crossings)
        detections = sorted(np.asarray(anomaly_indexes).astype(int).tolist())

        # =====================================================
        # Match detections to crossings
        # =====================================================
        used = set()
        matches = []

        for cross in true_crossings:

            candidates = []

            for det in detections:
                if det in used:
                    continue

                dist = abs(det - cross)

                if dist <= match_window:
                    candidates.append((dist, det))

            if candidates:
                _, best_det = min(candidates, key=lambda x: x[0])
                used.add(best_det)
                matches.append((cross, best_det))

        TP = len(matches)
        FP = len(detections) - TP
        FN = len(true_crossings) - TP

        precision = TP / (TP + FP) if (TP + FP) else 0.0
        recall = TP / (TP + FN) if (TP + FN) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        avg_distance_error = 0
        for True_crossing, Det in matches:
            diff_long = longitude[Det] - longitude[int(np.floor(True_crossing))]
            diff_lat = latitude[Det] - latitude[int(np.floor(True_crossing))]
            avg_distance_error += haversine_distance(latitude[Det], longitude[Det], latitude[int(np.floor(True_crossing))], longitude[int(np.floor(True_crossing))])
        if(len(matches) > 0):
            avg_distance_error /= len(matches)
        # =====================================================
        # Plot
        # =====================================================
        plt.figure(figsize=(10, 8))

        sc = plt.scatter(
            longitude,
            latitude,
            c=sig_mag,
            cmap="viridis",
            s=18,
            alpha=0.8,
        )

        # Start / End
        plt.scatter(
            longitude[0],
            latitude[0],
            marker="o",
            c="green",
            s=70,
            label="Start",
            zorder=5,
        )

        plt.scatter(
            longitude[-1],
            latitude[-1],
            marker="X",
            c="blue",
            s=70,
            label="End",
            zorder=5,
        )

        # -----------------------------------------------------
        # Plot cable segments
        # -----------------------------------------------------
        for k, (lon1, lat1, lon2, lat2) in enumerate(cable_segments):
            plt.plot(
                [lon1, lon2],
                [lat1, lat2],
                "--",
                lw=2,
                color="black",
                alpha=0.8,
                label="Cable" if k == 0 else None,
            )

        # -----------------------------------------------------
        # Plot detections
        # -----------------------------------------------------
        if len(detections) > 0:
            plt.scatter(
                longitude[detections],
                latitude[detections],
                marker="x",
                c="red",
                s=60,
                linewidths=2,
                label="Detected anomalies",
                zorder=6,
            )

        # -----------------------------------------------------
        # Plot true crossings
        # -----------------------------------------------------
        cross_x = []
        cross_y = []

        for c in true_crossings:
            i = int(np.floor(c))
            x = (longitude[i] + longitude[i + 1]) / 2
            y = (latitude[i] + latitude[i + 1]) / 2
            cross_x.append(x)
            cross_y.append(y)

        if len(cross_x):
            plt.scatter(
                cross_x,
                cross_y,
                marker="*",
                c="gold",
                edgecolors="black",
                s=160,
                label="True crossings",
                zorder=7,
            )

        # -----------------------------------------------------
        # Draw matching lines
        # -----------------------------------------------------
        for cross, det in matches:

            i = int(np.floor(cross))
            cx = (longitude[i] + longitude[i + 1]) / 2
            cy = (latitude[i] + latitude[i + 1]) / 2

            dx = longitude[det]
            dy = latitude[det]

            plt.plot(
                [cx, dx],
                [cy, dy],
                lw=1.8,
                color="black",
                alpha=0.9,
            )

        # =====================================================
        # Labels
        # =====================================================
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Trajectory / Cable Crossing Detection Metrics")
        plt.axis("equal")
        plt.grid(True)

        cbar = plt.colorbar(sc)
        cbar.set_label("Magnetic anomaly magnitude (µT)")

        # =====================================================
        # Metrics box
        # =====================================================
        txt = (
            f"Correct detections = {TP}\n"
            f"False detections = {FP}\n"
            f"Missed detections = {FN}\n"
            f"Avg. Distance Error = {avg_distance_error:.2f} m"
        )

        plt.gca().text(
            0.02,
            0.98,
            txt,
            transform=plt.gca().transAxes,
            va="top",
            bbox=dict(boxstyle="round", alpha=0.9),
            fontsize=10,
        )

        if caption:
            plt.figtext(
                0.5,
                0.02,
                caption,
                ha="center",
                fontsize=9,
            )

        plt.legend()
        plt.tight_layout()
        plt.show(block=block)

        return {
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "matches": matches,
            "true_crossings": true_crossings,
        }
        
    @staticmethod
    @paper_style
    def plot_filter_outputs(
        output_signal, nb_basis, block=False, confidence=None, caption=""
    ):
        """Plot filter outputs and a merged signal summary.

        Parameters
        ----------
        output_signal : array-like, shape (nb_basis, 3, N)
            The filter outputs to plot.
        nb_basis : int
            Number of filter basis plots to draw.
        block : bool, optional
            If True, block execution until the figure window is closed.
        confidence : float or array-like, optional
            Confidence indicator drawn on the merged output subplot.
        caption : str, optional
            Optional caption text displayed at the bottom of the figure.

        Returns
        -------
        None
        """
        plt.figure(figsize=(12, 3 * (nb_basis + 1)))
        print(">>>>> plot function >>>>> output shape", output_signal.shape)
        for i in range(nb_basis):
            plt.subplot(nb_basis + 1, 1, i + 1)
            plt.plot(
                output_signal[i].T,
                label=["x", "y", "z"] if nb_basis > 2 else "norm",
            )
            plt.title(f"Filter Output {i + 1}")
            plt.xlabel("samples")
            plt.ylabel("Amplitude")
            plt.grid()
            plt.legend()
        plt.subplot(nb_basis + 1, 1, nb_basis + 1)
        merged_output = np.linalg.norm(output_signal, axis=(0, 1))
        plt.plot(merged_output, label="Merged Output", color="purple")
        plt.title("Merged Filter Output")
        plt.xlabel("samples")
        plt.ylabel("Amplitude")

        N = merged_output.shape[0]
        margin = int(0.3 * N)
        center = merged_output[margin:-margin]
        center_max = np.max(center)
        center_min = np.min(center)
        diff = center_max - center_min
        plt.ylim(center_min - 2 * diff, center_max + 2 * diff)

        if confidence is not None:
            plt.plot(
                confidence * np.max(merged_output),
                label="Low Confidence Regions",
                color="red",
                alpha=0.3,
            )

        plt.grid()
        plt.legend()

        if caption:
            plt.subplots_adjust(bottom=0.18)
            plt.gcf().text(0.5, 0.04, caption, ha="center", fontsize=9)

        plt.tight_layout()
        plt.show(block=block)

    @staticmethod
    @paper_style
    def plot_anomaly_points(
        anom_time,
        rx_merged,
        anomaly_indexes=np.array([]),
        anom_mag_calib=None,
        true_cable_crossings=None,
        block=False,
        caption="",
    ):
        """
        Plot matched filter output, detected anomalies, and optional normalized
        anomaly magnitude calibration.

        Parameters
        ----------
        anom_time : array-like
            Time values corresponding to the matched filter output.
        rx_merged : array-like
            Matched filter output values to plot.
        anomaly_indexes : array-like, optional
            Indexes of detected anomalies to highlight.
        anom_mag_calib : array-like, optional
            Anomaly magnitude calibration signal to normalize and overlay.
        block : bool, optional
            If True, block execution until the plot window is closed.
        caption : str, optional
            Caption text to display at the bottom of the figure.

        Returns
        -------
        None
        """
        anom_time = np.asarray(anom_time)
        rx_merged = np.asarray(rx_merged)
        anomaly_indexes = np.asarray(anomaly_indexes)

        plt.figure()
        plt.plot(anom_time, rx_merged, label="Normalized filter output", alpha=0.8)

        if anomaly_indexes.size > 0:
            plt.plot(
                anom_time[anomaly_indexes],
                rx_merged[anomaly_indexes],
                "ro",
                label="Detected Anomalies",
                zorder=5,
            )

        if anom_mag_calib is not None and anomaly_indexes.size > 0:
            norm_calib = Normalize(anom_mag_calib)
            plt.plot(
                anom_time,
                norm_calib,
                label="Normalized Anomaly Magnitude",
                alpha=0.5,
            )
        
        if true_cable_crossings is not None and len(true_cable_crossings) > 0:
            plt.plot(
                anom_time[true_cable_crossings],
                rx_merged[true_cable_crossings],
                "kx",
                label="True Cable Crossings",
                zorder=5,
            )

        plt.title("Filtered Signal with Detected Anomalies")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.legend()

        if caption:
            plt.subplots_adjust(bottom=0.18)
            plt.gcf().text(0.5, 0.04, caption, ha="center", fontsize=9)

        plt.tight_layout()
        plt.show(block=block)

    @staticmethod
    @paper_style
    def plot_anomalies_with_velocity(
        time,
        anom_mag_vec,
        anom_mag_norm,
        vel_vec,
        vel_norm,
        block=False,
        confidence=None,
        confidence_thresholds=None,
        caption="",
    ):
        """
        Plot anomalies in magnetic field and drone velocity over time.

        Parameters
        ----------
        time : array-like
            Time values corresponding to the measurements.
        anom_mag_vec : array-like, shape (3, N)
            Magnetic field anomaly vector components (X, Y, Z) over time.
        anom_mag_norm : array-like, shape (N,)
            Magnitude of the magnetic field anomaly over time.
        vel_vec : array-like, shape (3, N)
            Drone velocity vector components (forward, right, down) over time.
        vel_norm : array-like, shape (N,)
            Magnitude of the drone velocity over time.
        block : bool, optional
            If True, block execution until the plot window is closed. Default is False.
        confidence : array-like or None, optional
            Confidence values to overlay on the velocity plot. Default is None.
        confidence_thresholds : tuple or list or None, optional
            Threshold values for confidence. Default is None.
        caption : str, optional
            Caption text to display at the bottom of the figure. Default is "".

        Returns
        -------
        None
        """
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(
            time, anom_mag_vec[0, :], label="Magnetic Field X Component", alpha=0.5
        )
        plt.plot(
            time, anom_mag_vec[1, :], label="Magnetic Field Y Component", alpha=0.5
        )
        plt.plot(
            time, anom_mag_vec[2, :], label="Magnetic Field Z Component", alpha=0.5
        )
        plt.plot(time, anom_mag_norm, label="Magnetic Field Magnitude", color="black")
        plt.title("Anomalous Magnetic Field Magnitude over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Magnetic Field Magnitude (nT)")
        plt.grid()
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(time, vel_vec[0, :], label="Drone Forward Velocity", alpha=0.5)
        plt.plot(time, vel_vec[1, :], label="Drone Right Velocity", alpha=0.5)
        plt.plot(time, vel_vec[2, :], label="Drone Down Velocity", alpha=0.5)
        plt.plot(time, vel_norm, label="Drone Velocity Magnitude", color="black")

        if confidence is not None:
            plt.plot(
                time,
                confidence * np.max(vel_norm),
                label="Low Confidence Regions",
                color="red",
                alpha=0.3,
            )
            if confidence_thresholds is not None:
                plt.hlines(
                    confidence_thresholds[0],
                    time[0],
                    time[-1],
                    colors="orange",
                    linestyles="dashed",
                    label="Confidence Thresholds",
                )
                plt.hlines(
                    confidence_thresholds[1],
                    time[0],
                    time[-1],
                    colors="orange",
                    linestyles="dashed",
                )

        plt.title("Drone Velocity Magnitude over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity Magnitude (m/s)")
        plt.grid()
        plt.legend()

        if caption:
            plt.subplots_adjust(bottom=0.18)
            plt.gcf().text(0.5, 0.04, caption, ha="center", fontsize=9)

        plt.tight_layout()
        plt.show(block=block)

    @staticmethod
    @paper_style
    def plot_anomalies_with_cap(
        time,
        anom_mag_vec,
        anom_mag_norm,
        cap,
        block=False,
        confidence=None,
        confidence_thresholds=None,
        caption="",
    ):
        """
        Plot anomalies in magnetic field and CAP signal over time.

        Parameters
        ----------
        time : array-like
            Time values.
        anom_mag_vec : array-like, shape (3, N)
            Magnetic field anomaly vector.
        anom_mag_norm : array-like, shape (N,)
            Magnetic field anomaly magnitude.
        cap : array-like, shape (N,)
            CAP signal to display on the second subplot.
        block : bool, optional
            If True, block execution until the plot window is closed.
        confidence : array-like or None, optional
            Confidence values to overlay. Default is None.
        confidence_thresholds : tuple or None, optional
            Threshold values for confidence. Default is None.
        caption : str, optional
            Caption text. Default is "".

        Returns
        -------
        None
        """
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(
            time, anom_mag_vec[0, :], label="Magnetic Field X Component", alpha=0.5
        )
        plt.plot(
            time, anom_mag_vec[1, :], label="Magnetic Field Y Component", alpha=0.5
        )
        plt.plot(
            time, anom_mag_vec[2, :], label="Magnetic Field Z Component", alpha=0.5
        )
        plt.plot(time, anom_mag_norm, label="Magnetic Field Magnitude", color="black")
        plt.title("Anomalous Magnetic Field Magnitude over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Magnetic Field Magnitude (nT)")
        plt.grid()
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(time, cap, label="Drone Velocity Magnitude", color="black")

        if confidence is not None:
            plt.plot(
                time,
                confidence * np.max(cap),
                label="Low Confidence Regions",
                color="red",
                alpha=0.3,
            )
            if confidence_thresholds is not None:
                plt.hlines(
                    confidence_thresholds[0],
                    time[0],
                    time[-1],
                    colors="orange",
                    linestyles="dashed",
                    label="Confidence Thresholds",
                )
                plt.hlines(
                    confidence_thresholds[1],
                    time[0],
                    time[-1],
                    colors="orange",
                    linestyles="dashed",
                )

        plt.title("Drone Velocity Magnitude over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity Magnitude (m/s)")
        plt.grid()
        plt.legend()

        if caption:
            plt.subplots_adjust(bottom=0.18)
            plt.gcf().text(0.5, 0.04, caption, ha="center", fontsize=9)

        plt.tight_layout()
        plt.show(block=block)

    @staticmethod
    @paper_style
    def plot_anderson_basis(basis_functions, block=False, caption=""):
        """
        Plot all Anderson basis functions on a single axes.

        Parameters
        ----------
        basis_functions : array-like, shape (nb_basis, N)
            The Anderson basis functions to plot.
        block : bool, optional
            If True, block execution until the plot window is closed. Default is False.
        caption : str, optional
            Caption text to display at the bottom of the figure. Default is "".

        Returns
        -------
        None
        """
        bf = np.asarray(basis_functions)
        if bf.ndim == 1:
            bf = bf[np.newaxis, :]

        nb_basis = bf.shape[0]
        plt.figure(figsize=(12, 6))
        for i in range(nb_basis):
            plt.plot(bf[i], label=f"Anderson Basis {i + 1}", linewidth=1)

        plt.title("Anderson Basis Functions (all in one plot)")
        plt.xlabel("samples")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend(ncol=min(4, nb_basis))

        if caption:
            plt.subplots_adjust(bottom=0.18)
            plt.gcf().text(0.5, 0.04, caption, ha="center", fontsize=9)

        plt.tight_layout()
        plt.show(block=block)

    @staticmethod
    @paper_style
    def plot_spectrum(magvec, fs, block=False, caption=""):
        """
        Plot the single-sided amplitude spectrum of the 3-axis magnetometer signal.

        Parameters
        ----------
        magvec : np.ndarray, shape (3, N)
            Magnetometer signal with rows [X, Y, Z].
        fs : float
            Sampling frequency in Hz.
        block : bool, optional
            If True, block execution until the plot window is closed. Default is False.
        caption : str, optional
            Caption text to display at the bottom of the figure. Default is "".

        Returns
        -------
        None
        """
        n = magvec.shape[1]
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        mask = freqs > 0

        labels = ["X", "Y", "Z"]
        colors = ["tab:blue", "tab:orange", "tab:green"]

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle("Signal Spectrum (positive frequencies, log scale)", fontsize=14)

        for ax, label, color, row in zip(axes, labels, colors, magvec):
            spectrum = np.abs(np.fft.rfft(row)) / n
            spectrum[1:-1] *= 2

            ax.plot(freqs[mask], spectrum[mask], color=color, linewidth=0.8)
            ax.set_ylabel(f"Mag {label}\nAmplitude", fontsize=10)
            ax.set_xscale("log")
            ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.7)
            ax.set_xlim(left=freqs[mask][0], right=fs / 2)

        axes[-1].set_xlabel("Frequency [Hz]", fontsize=11)

        if caption:
            plt.subplots_adjust(bottom=0.18)
            fig.text(0.5, 0.04, caption, ha="center", fontsize=9)

        plt.tight_layout()
        plt.show(block=block)
