"""Helpers for the multi-candidate transit path model (--num_path)."""

from src.constants import S_SET


def path_time(path, freq, w_speed):
    """Total transit time [min] of `path` at frequencies `freq`; inf if a line is not operated."""
    wait = 0.0
    for l in path['gamma_odl']:
        s = freq[l]
        if s <= 1e-6:
            return float('inf')
        wait += 30.0 / s
    return path['tau_pt'] + path['walking_dist'] / w_speed + wait


def path_time_bounds(path, w_speed):
    """(t_lo, t_up): path_time at the highest resp. lowest frequency in S_SET on every line."""
    fixed = path['tau_pt'] + path['walking_dist'] / w_speed
    n = len(path['gamma_odl'])
    return fixed + n * 30.0 / max(S_SET), fixed + n * 30.0 / min(S_SET)


def choose_path(paths, freq, w_speed, prefer=None):
    """(index, time) of the min-time path; keep `prefer` when it ties the minimum within 1e-6."""
    times = [path_time(p, freq, w_speed) for p in paths]
    t_min = min(times)
    if t_min == float('inf'):
        return (prefer if prefer is not None else 0), t_min
    if prefer is not None and times[prefer] <= t_min + 1e-6:
        return prefer, times[prefer]
    return times.index(t_min), t_min


def segment_incidence(pt_paths):
    """{(u, v, line_id): [(od, k), ...]} listing the candidate paths using each transit segment."""
    inc = {}
    for od, paths in pt_paths.items():
        for k, p in enumerate(paths):
            for seg in p['eta_e_od']:
                inc.setdefault(seg, []).append((od, k))
    return inc
