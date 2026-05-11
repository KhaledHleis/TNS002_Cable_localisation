from pathlib import Path
from datetime import datetime
import pickle
import sys
from matplotlib.figure import Figure


def prepare_experiment(base_dir="experiments", prompt="Experiment name: "):
    """
    Ask for an experiment name, replace '#' with timestamp,
    and create the directory:

        experiments/<script_name>/<experiment_name>/

    Returns
    -------
    Path
        Path to the experiment directory
    """
    # ---- get experiment name ----
    name = input(prompt).strip()

    timestamp = datetime.now().strftime("%H_%M_%d_%m_%y")
    name = name.replace("#", timestamp)
    if name == "":
        print("No experiment name provided. Skipping directory creation.")
        return None
    else:
        # ---- infer script name ----
        script_path = Path(sys.argv[0])
        script_name = script_path.stem if script_path.name else "interactive"

        # ---- build experiment path ----
        base_path = Path(base_dir) / script_name
        exp_path = base_path / name

        if not exp_path.exists():
            exp_path.mkdir(parents=True)
            return exp_path

        i = 1
        while True:
            new_exp_path = base_path / f"{name}_{i}"
            if not new_exp_path.exists():
                new_exp_path.mkdir(parents=True)
                print(f"Experiment directory created: {new_exp_path.resolve()}")
                return new_exp_path
            i += 1


def save_figure_pickle(fig, exp_dir, prefix="fig"):
    """
    Pickle a matplotlib figure into the experiment directory.

    - Filename is taken from the figure title if available
    - Otherwise a timestamp-based name is used
    - If the file already exists, a number is appended to keep both files
    """
    if not isinstance(fig, Figure):
        raise TypeError("Expected a matplotlib.figure.Figure")
    if exp_dir is None:
        print("No experiment directory provided. Skipping figure saving.")
        return None
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # ---- get figure title ----
    title = fig.axes[0].get_title() if fig.axes else None

    if title:
        base_name = title.strip().replace(" ", "_")
    else:
        timestamp = datetime.now().strftime("%H_%M_%d_%m_%y")
        base_name = f"{prefix}_{timestamp}"

    # ---- resolve filename collisions ----
    file_path = exp_dir / f"{base_name}.pickle"
    counter = 1
    while file_path.exists():
        file_path = exp_dir / f"{base_name}_{counter}.pickle"
        counter += 1

    # ---- save figure ----
    fig.savefig(exp_dir / f"{base_name}_{counter}.png")
    with open(file_path, "wb") as f:
        pickle.dump(fig, f)
    fig.savefig(exp_dir / f"{base_name}_{counter}.png")
    print(f"Figure pickled: {file_path.resolve()}")
    return file_path
