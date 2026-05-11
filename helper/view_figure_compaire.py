import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

args = sys.argv[1:]

# Check for --normalize flag
normalize = "--normalize" in args
if normalize:
    args = [a for a in args if a != "--normalize"]

if len(args) % 2 != 0:
    print("Usage: python compare_figs.py fig1.pkl name1 fig2.pkl name2 ... [--normalize]")
    sys.exit(1)

pairs = [(args[i], args[i+1]) for i in range(0, len(args), 2)]

figures = []
for path, name in pairs:
    with open(path, "rb") as f:
        fig = pickle.load(f)
        figures.append((fig, name))

# Map title -> list of (ax, name) across all figures
axes_by_title = defaultdict(list)
for fig, name in figures:
    for ax in fig.axes:
        title = ax.get_title()
        if title:
            axes_by_title[title].append((ax, name))

# Keep only titles present in ALL figures
n_figs = len(figures)
mutual_titles = [t for t, axs in axes_by_title.items() if len(axs) == n_figs]

if not mutual_titles:
    print("No mutual subplot titles found across all figures.")
    sys.exit(1)

print(f"Mutual subplots: {mutual_titles}")


def normalize_signal(y):
    """Zero-mean then scale to [-1, 1]."""
    y = np.array(y, dtype=float)
    y = y - y.mean()
    peak = np.abs(y).max()
    if peak > 0:
        y = y / peak
    return y


# Create combined figure
combined_fig, combined_axes = plt.subplots(len(mutual_titles), 1, figsize=(10, 4 * len(mutual_titles)))

if len(mutual_titles) == 1:
    combined_axes = [combined_axes]

for ax_idx, title in enumerate(mutual_titles):
    ax = combined_axes[ax_idx]
    ref_ax = axes_by_title[title][0][0]

    ax.set_title(title + (" (normalized)" if normalize else ""))
    ax.set_xlabel(ref_ax.get_xlabel())
    ax.set_ylabel("Amplitude" if normalize else ref_ax.get_ylabel())

    for src_ax, name in axes_by_title[title]:
        for line in src_ax.get_lines():
            x = line.get_xdata()
            y = line.get_ydata()

            if normalize:
                y = normalize_signal(y)

            orig_label = line.get_label()
            if orig_label.startswith("_"):
                orig_label = ""
            label = f"{orig_label} --{name}" if orig_label else f"--{name}"

            ax.plot(
                x,
                y,
                label=label,
                linestyle=line.get_linestyle(),
                marker=line.get_marker()
            )

    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()