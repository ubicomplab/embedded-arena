"""Per-cycle energy and latency calculations from PPK2 current samples.

Consumes the dict returned by `ppk2.ppk2Monitor.run()` (keys
'currents' and 'cycles') and produces aggregated metrics that downstream
checks/tools can use as scores or feedback.
"""

import numpy as np


def calculate_metrics(ppk_data, sample_rate_hz=100_000, v_supply=3.3, skip_initial_cycles=2):
    """Compute per-cycle energy and current statistics from PPK data.

    The first skip_initial_cycles cycles are discarded because the firmware
    typically exhibits higher current during initialisation/warmup and would
    skew the averages.

    Args:
        ppk_data: Dict with 'currents' (array, µA) and 'cycles' (list of [start_idx, end_idx]).
        sample_rate_hz: PPK2 sampling rate in Hz.
        v_supply: Supply voltage in Volts (used for energy calculation).
        skip_initial_cycles: Number of leading cycles to ignore.

    Returns:
        Dict with avg_energy_j, avg_max_uA, avg_avg_uA, avg_time_s, used_cycles,
        or None if there are not enough valid cycles.
    """
    currents = ppk_data.get("currents", [])
    cycles = ppk_data.get("cycles", [])

    if cycles is None or len(cycles) <= skip_initial_cycles:
        return None

    per_cycle_avg    = []
    per_cycle_max    = []
    per_cycle_time_s = []
    per_cycle_energy = []

    for start, end in cycles[skip_initial_cycles:]:
        start_i = int(max(0, start))
        end_i   = int(min(len(currents), end))
        if end_i <= start_i:
            continue

        segment = currents[start_i:end_i]
        avg_uA  = float(np.mean(segment))
        max_uA  = float(np.max(segment))
        dt_s    = (end_i - start_i) / float(sample_rate_hz)

        # E(J) = I(A) × V(V) × t(s),  where I(A) = avg_µA × 1e-6
        energy_j = (avg_uA * 1e-6) * v_supply * dt_s

        per_cycle_avg.append(avg_uA)
        per_cycle_max.append(max_uA)
        per_cycle_time_s.append(dt_s)
        per_cycle_energy.append(energy_j)

    if not per_cycle_energy:
        return None

    return {
        "avg_energy_j": float(np.mean(per_cycle_energy)),
        "avg_max_uA":   float(np.mean(per_cycle_max)),
        "avg_avg_uA":   float(np.mean(per_cycle_avg)),
        "avg_time_s":   float(np.mean(per_cycle_time_s)),
        "used_cycles":  len(per_cycle_energy),
    }
