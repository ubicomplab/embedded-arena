"""Plotting utilities for hardware-in-the-loop iteration analysis.

Isolated here so callers can produce current-profile and per-iteration trend
figures without pulling matplotlib into modules that may run headless. Each
function saves one or more PNG files and closes figures before returning to
prevent memory leaks across iterations.
"""

import os
import matplotlib.pyplot as plt


def plot_current_profile(ppk_data, iteration, out_dir, extra_paths=None, sample_rate_hz=100_000):
    """Save a time-domain current-profile plot for one iteration.

    Detected active cycles are highlighted as red spans so the agent and user
    can visually verify cycle detection quality.

    Args:
        ppk_data: Dict with 'currents' (list/array) and 'cycles' (list of [start, end]).
        iteration: Iteration number used in the plot title and filename.
        out_dir: Primary directory where the file is written.
        extra_paths: Optional list of additional paths to receive copies of the figure.
        sample_rate_hz: PPK2 sampling rate used to convert sample indices to wall-clock seconds.

    Returns:
        Primary output path, or None if ppk_data carries no current samples.
    """
    currents = ppk_data.get("currents", [])
    cycles = ppk_data.get("cycles", [])
    if len(currents) == 0:
        return None

    os.makedirs(out_dir, exist_ok=True)
    t_axis = [i / sample_rate_hz for i in range(len(currents))]

    plt.figure(figsize=(12, 6))
    plt.plot(t_axis, currents, label="Current (uA)", linewidth=0.8)
    for i, (start, end) in enumerate(cycles):
        plt.axvspan(
            start / sample_rate_hz,
            end / sample_rate_hz,
            color="red",
            alpha=0.3,
            label="cycle" if i == 0 else "",
        )
    plt.title(f"Current Profile - Iteration {iteration}")
    plt.xlabel("Time (s)")
    plt.ylabel("Current (uA)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    filename = f"current_plot_iter_{iteration}.png"
    primary = os.path.join(out_dir, filename)
    plt.savefig(primary)
    for path in extra_paths or []:
        plt.savefig(path)
    plt.close()
    return primary


def plot_iter_stats(iter_history, iteration, out_dir, extra_paths=None):
    """Save a combined 4-panel stats scatter and separate energy / latency trend plots.

    Three files are written per call:
      - iter_stats_scatter_iter_<N>.png  (4-panel combined)
      - energy_iter_<N>.png
      - latency_iter_<N>.png

    Args:
        iter_history: Dict with parallel lists keyed by
            'iters', 'energy_j', 'max_uA', 'avg_uA', 'time_s'.
        iteration: Current iteration number (used in filenames).
        out_dir: Primary output directory.
        extra_paths: Dict mapping plot type ('combined', 'energy', 'latency')
            to a list of additional save paths for that figure.
    """
    iters = iter_history["iters"]
    energy_j = iter_history["energy_j"]
    max_uA = iter_history["max_uA"]
    avg_uA = iter_history["avg_uA"]
    time_s = iter_history["time_s"]
    extra = extra_paths or {}

    os.makedirs(out_dir, exist_ok=True)

    # Combined 4-panel scatter
    fig, axes = plt.subplots(4, 1, figsize=(8, 12), sharex=True)
    panels = [
        (axes[0], energy_j, "Avg Cycle Energy (3.3V)",      "Energy (J)",       "steelblue"),
        (axes[1], max_uA,   "Avg Max Current per Iteration", "Max Current (uA)", "steelblue"),
        (axes[2], avg_uA,   "Avg Current per Iteration",     "Avg Current (uA)", "orange"),
        (axes[3], time_s,   "Avg Duration per Iteration",    "Time (s)",         "green"),
    ]
    for ax, data, title, ylabel, color in panels:
        ax.plot(iters, data, linestyle="--", color="gray", alpha=0.5)
        ax.scatter(iters, data, s=30, color=color)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Iteration")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"iter_stats_scatter_iter_{iteration}.png"))
    for p in extra.get("combined", []):
        plt.savefig(p)
    plt.close()

    # Energy-only trend
    plt.figure(figsize=(8, 4))
    plt.plot(iters, energy_j, linestyle="--", color="gray", alpha=0.5)
    plt.scatter(iters, energy_j, s=40, zorder=5)
    plt.title("Avg Cycle Energy across Iterations (3.3V)")
    plt.xlabel("Iteration")
    plt.ylabel("Energy (J)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"energy_iter_{iteration}.png"))
    for p in extra.get("energy", []):
        plt.savefig(p)
    plt.close()

    # Latency-only trend
    plt.figure(figsize=(8, 4))
    plt.plot(iters, time_s, linestyle="--", color="gray", alpha=0.5)
    plt.scatter(iters, time_s, s=40, color="green", zorder=5)
    plt.title("Avg Cycle Latency across Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Latency (s)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"latency_iter_{iteration}.png"))
    for p in extra.get("latency", []):
        plt.savefig(p)
    plt.close()
