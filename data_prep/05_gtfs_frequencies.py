"""GTFS AM-peak line frequencies: baseline (no-subsidy) frequencies for baseline_amod_br.py.

Counts the trips starting in the AM-peak window of the busiest weekday service, converts to trips per
hour and snaps to the frequency set S_SET. Lines are those of metro/optimization_inputs.pkl.gz (run 04
first). Writes metro/gtfs_frequencies.csv (details) and metro/gtfs_frequencies.pkl ({line_id: trains/hour}).
"""
import argparse
import gzip
import os
import pickle
import sys
import pandas as pd
from common import GTFS_NAME

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.constants import S_SET

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--raw_dir', default='data_nyc')
parser.add_argument('--out_dir', default='data_nyc/processed')
parser.add_argument('--window_start', default='07:00:00')
parser.add_argument('--window_end', default='10:00:00')
args = parser.parse_args()

start, end = pd.to_timedelta(args.window_start), pd.to_timedelta(args.window_end)
hours = (end - start).total_seconds() / 3600
DIRECTION = {0: 'N', 1: 'S'}

with gzip.open(f'{args.out_dir}/metro/optimization_inputs.pkl.gz', 'rb') as f:
    lines = list(pickle.load(f)['lines'])

gtfs = f'{args.raw_dir}/{GTFS_NAME}'
trips = pd.read_csv(f'{gtfs}/trips.txt')
stop_times = pd.read_csv(f'{gtfs}/stop_times.txt')
calendar = pd.read_csv(f'{gtfs}/calendar.txt')

# busiest weekday service (Monday-running service ids with the most trips)
weekday_trips = trips[trips['service_id'].isin(calendar.loc[calendar['monday'] == 1, 'service_id'])]
service_id = weekday_trips.groupby('service_id').size().idxmax()
print(f"Weekday service_id: {service_id}")

# trips starting in the window, by departure time at their first stop
first_stop = stop_times.sort_values(['trip_id', 'stop_sequence']).drop_duplicates('trip_id')[['trip_id', 'departure_time']]
first_stop['dep'] = pd.to_timedelta(first_stop['departure_time'])
starts = first_stop.merge(trips[trips['service_id'] == service_id][['trip_id', 'route_id', 'direction_id']], on='trip_id')
peak = starts[(starts['dep'] >= start) & (starts['dep'] < end)]

counts = peak.groupby(['route_id', 'direction_id']).size().rename('trips_am_peak').reset_index()
counts['line_id'] = counts['route_id'].astype(str) + '_' + counts['direction_id'].map(DIRECTION)
counts = counts.set_index('line_id')['trips_am_peak']


def snap(v):
    """Nearest value in S_SET (ties to the lower one); 0 stays 0."""
    return min(S_SET, key=lambda s: (abs(s - v), s)) if v > 0 else 0


table = pd.DataFrame({'line_id': lines})
table['route_id'], table['direction'] = zip(*table['line_id'].str.rsplit('_', n=1))
table['trips_am_peak'] = table['line_id'].map(counts).fillna(0).astype(int)  # lines without AM-peak trips -> 0
table['freq_raw_vph'] = table['trips_am_peak'] / hours
table['freq_snapped_vph'] = table['freq_raw_vph'].apply(snap).astype(int)

print(f"{(table['trips_am_peak'] == 0).sum()} lines without AM-peak trips: {table.loc[table['trips_am_peak'] == 0, 'line_id'].tolist()}")
dropped = table[(table['freq_raw_vph'] > 0) & (table['freq_snapped_vph'] == 0)]
if not dropped.empty:
    print(f"WARNING: lines snapped from a positive frequency to 0:\n{dropped}")

table.to_csv(f'{args.out_dir}/metro/gtfs_frequencies.csv', index=False)
with open(f'{args.out_dir}/metro/gtfs_frequencies.pkl', 'wb') as f:
    pickle.dump(table.set_index('line_id')['freq_snapped_vph'].to_dict(), f)
