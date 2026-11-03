"""TLC FHV trips -> intra-Manhattan OD demand in a time window (processed/demand.csv)."""
import argparse
import os
import pandas as pd
from common import EXCLUDED_ZONES, load_manhattan_zones

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--raw_dir', default='data_nyc')
parser.add_argument('--out_dir', default='data_nyc/processed')
parser.add_argument('--trips_file', default='demand/fhvhv_tripdata_2024-05.parquet', help='relative to raw_dir')
parser.add_argument('--date', default='2024-05-15')
parser.add_argument('--start_hour', type=int, default=9, help='window is [start_hour, end_hour)')
parser.add_argument('--end_hour', type=int, default=11)
args = parser.parse_args()

trips = pd.read_parquet(f'{args.raw_dir}/{args.trips_file}', columns=['pickup_datetime', 'PULocationID', 'DOLocationID'])
t = pd.to_datetime(trips['pickup_datetime'])
window = trips[(t.dt.date == pd.Timestamp(args.date).date()) & t.dt.hour.between(args.start_hour, args.end_hour - 1)]
print(f"Trips {args.start_hour:02d}-{args.end_hour:02d}h on {args.date}: {len(window):,}")

zone_ids = set(load_manhattan_zones(args.raw_dir)['LocationID'])
od = window[window['PULocationID'].isin(zone_ids) & window['DOLocationID'].isin(zone_ids)]
od = od[od['PULocationID'] != od['DOLocationID']]  # drop intra-zone trips
demand = od.groupby(['PULocationID', 'DOLocationID']).size().reset_index(name='trip_count').sort_values('trip_count', ascending=False)
print(f"{len(demand):,} OD pairs, {demand['trip_count'].sum():,} trips (zones {len(zone_ids)}, excluded {EXCLUDED_ZONES})")

os.makedirs(args.out_dir, exist_ok=True)
demand.to_csv(f'{args.out_dir}/demand.csv', index=False)
