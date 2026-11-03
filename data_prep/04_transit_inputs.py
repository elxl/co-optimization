"""r5py itineraries -> transit optimization inputs.

Ranks the distinct subway paths of every OD by travel time (then walking distance), applies the path
and demand filters, and writes
  metro/optimization_inputs.pkl.gz     best path per OD (--num_path 1), representative-trip line topology
  metro/optimization_inputs_k5.pkl.gz  up to --k candidate paths per OD, line topology = union of all trip patterns
Path 0 of the k-path file is the path of the single-path file. The ODs kept are those whose best path
passes the filters, so both files have the same OD set. The two files keep different line topologies
for backward compatibility (the sweeps use the k-path file even for K=1).

Format: {"optimization_data": {(o, d): path or [paths]}, "lines": {line_id: [(stop_u, stop_v), ...]}}
with path = {walking_dist, tau_pt, gamma_odl (lines used), eta_e_od ({(stop_u, stop_v, line_id)})}.
"""
import argparse
import gzip
import pickle
from functools import lru_cache
import pandas as pd
from common import GTFS_NAME

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--raw_dir', default='data_nyc')
parser.add_argument('--out_dir', default='data_nyc/processed')
parser.add_argument('--k', type=int, default=5, help='max candidate paths per OD in the k-path file')
parser.add_argument('--max_walk', type=float, default=1200, help='max walking distance of a path (m)')
parser.add_argument('--max_legs', type=int, default=3, help='max number of subway legs of a path')
parser.add_argument('--min_trips', type=int, default=4, help='min demand (trips) of an OD pair')
args = parser.parse_args()

SUBWAY, WALK = 'TransportMode.SUBWAY', 'TransportMode.WALK'
DIRECTION = {0: 'N', 1: 'S'}  # GTFS direction_id -> suffix of the stop ids
OD = ['PULocationID', 'DOLocationID']

df = pd.read_csv(f'{args.out_dir}/metro/itineraries_raw_multi.csv.gz')
df['travel_time'] = pd.to_timedelta(df['travel_time'])
df['wait_time'] = pd.to_timedelta(df['wait_time'])
demand = pd.read_csv(f'{args.out_dir}/demand.csv')

gtfs = f'{args.raw_dir}/{GTFS_NAME}'
trips = pd.read_csv(f'{gtfs}/trips.txt')
stop_times = pd.read_csv(f'{gtfs}/stop_times.txt')


# ── Rank the distinct paths of every OD ───────────────────────────────────────
def path_signature(x):
    """Ordered (route, direction, boarding stop, alighting stop) of the subway legs: equal signature = same path."""
    legs = x[x['transport_mode'] == SUBWAY].sort_values('segment')
    return tuple((r.route_id, str(r.start_stop_id)[-1], r.start_stop_id, r.end_stop_id) for r in legs.itertuples())


def path_metrics(x):
    walk = x['transport_mode'] == WALK
    return pd.Series({
        'total_duration_wo_waiting': x['travel_time'].sum().total_seconds() / 60,
        'walking_dist': x.loc[walk, 'distance'].sum(),
        'walking_time': x.loc[walk, 'travel_time'].sum().total_seconds() / 60,
        'transit_legs': (x['transport_mode'] == SUBWAY).sum(),
    })


options = df.groupby(OD + ['option'])
paths = options.apply(path_metrics, include_groups=False).join(
    options.apply(path_signature, include_groups=False).rename('signature')).reset_index()
paths = paths.sort_values(OD + ['total_duration_wo_waiting', 'walking_dist'])
paths = paths.drop_duplicates(OD + ['signature']).groupby(OD).head(args.k)  # top-k distinct paths per OD

# ── Filters: per-path walking / legs; per-OD best path and demand ─────────────
ok = (paths['walking_dist'] <= args.max_walk) & paths['transit_legs'].between(1, args.max_legs)
best_ok = paths[ok & (paths.groupby(OD).cumcount() == 0)].merge(demand[demand['trip_count'] >= args.min_trips][OD])[OD]
paths = paths[ok].merge(best_ok)  # drops the ODs whose best path fails
paths['k_idx'] = paths.groupby(OD).cumcount()
paths['tau_pt'] = paths['total_duration_wo_waiting'] - paths['walking_time']  # in-vehicle + transfer time, no waiting
print(f"{len(best_ok)} ODs kept; candidates per OD: {paths.groupby(OD).size().value_counts().sort_index().to_dict()}")


# ── Path parameters: lines used (gamma) and directed segments used (eta) ──────
@lru_cache(maxsize=None)
def route_stop_times(route_id):
    return stop_times[stop_times['trip_id'].isin(trips.loc[trips['route_id'] == route_id, 'trip_id'])]


@lru_cache(maxsize=None)
def leg_segments(route_id, start, end):
    """Directed stop-to-stop segments from boarding to alighting stop, on the first trip of the route serving the boarding stop."""
    rel = route_stop_times(route_id)
    try:
        t = rel[rel['trip_id'] == rel.loc[rel['stop_id'] == start, 'trip_id'].iloc[0]]
        lo = t.loc[t['stop_id'] == start, 'stop_sequence'].iloc[0]
        hi = t.loc[t['stop_id'] == end, 'stop_sequence'].iloc[0]
    except IndexError:  # alighting stop not on that trip
        return ()
    stops = t[t['stop_sequence'].between(lo, hi)].sort_values('stop_sequence')['stop_id'].tolist()
    return tuple(zip(stops[:-1], stops[1:]))


params = {(int(r.PULocationID), int(r.DOLocationID), r.k_idx):
          {'walking_dist': float(r.walking_dist), 'tau_pt': float(r.tau_pt), 'gamma_odl': set(), 'eta_e_od': set()}
          for r in paths.itertuples()}
legs = df[df['transport_mode'] == SUBWAY].merge(paths[OD + ['option', 'k_idx']], on=OD + ['option'])
for r in legs.itertuples():
    p = params[(int(r.PULocationID), int(r.DOLocationID), r.k_idx)]
    line = f"{r.route_id}_{r.start_stop_id[-1]}"
    p['gamma_odl'].add(line)
    p['eta_e_od'].update((u, v, line) for u, v in leg_segments(r.route_id, r.start_stop_id, r.end_stop_id))

data_k = {}  # (o, d) -> [path_0, path_1, ...] in k_idx order
for (o, d, _), p in sorted(params.items()):
    data_k.setdefault((o, d), []).append(p)
data_1 = {od: ps[0] for od, ps in data_k.items()}


# ── Line topology: consecutive-stop segments of every (route, direction) ──────
stops_by_trip = {t: g.sort_values('stop_sequence')['stop_id'].tolist() for t, g in stop_times.groupby('trip_id')}


def line_topology(union_patterns):
    """union_patterns=False: one representative trip per line (can be a short-turn or late-night pattern);
    True: union over all trips of the line, so every segment a path traverses belongs to its line."""
    lines = {}
    for (route, direction), g in trips.groupby(['route_id', 'direction_id']):
        trip_ids = list(g['trip_id']) if union_patterns else [g['trip_id'].iloc[0]]
        segments = {}  # dict as ordered set
        for t in trip_ids:
            s = stops_by_trip.get(t, [])
            segments.update(dict.fromkeys(zip(s[:-1], s[1:])))
        lines[f"{route}_{DIRECTION.get(direction, str(direction))}"] = list(segments)
    return lines


for name, data, lines in [('optimization_inputs', data_1, line_topology(False)),
                          (f'optimization_inputs_k{args.k}', data_k, line_topology(True))]:
    with gzip.open(f'{args.out_dir}/metro/{name}.pkl.gz', 'wb') as f:
        pickle.dump({'optimization_data': data, 'lines': lines}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"{name}: {len(data)} ODs, {len(lines)} lines")
