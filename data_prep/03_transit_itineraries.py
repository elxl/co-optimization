"""r5py transit + walk itineraries between zone centroids for every OD pair in demand.csv.

Writes the full multi-option r5py output (geometry dropped) to <out_dir>/metro/itineraries_raw_multi.csv.gz.
r5py is not bit-reproducible: a re-run picks the same best option for ~99.8% of ODs, with walking
distances differing by metres, so a re-run can move a few ODs across the filters in 04.
Needs Java (r5py) and the walking network from 02_clip_osm.sh.
"""
import argparse
import datetime
import os
import geopandas as gpd
import pandas as pd
import r5py
from common import GTFS_NAME, load_manhattan_zones, zone_centroids

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--raw_dir', default='data_nyc')
parser.add_argument('--out_dir', default='data_nyc/processed')
parser.add_argument('--osm_file', default='manhattan.osm.pbf', help='relative to raw_dir')
parser.add_argument('--departure', default='2026-05-15T09:00', help='must lie inside the GTFS calendar validity (calendar.txt)')
parser.add_argument('--max_hours', type=float, default=2)
args = parser.parse_args()

demand = pd.read_csv(f'{args.out_dir}/demand.csv')
zones = zone_centroids(load_manhattan_zones(args.raw_dir))[['LocationID', 'geometry']]


def od_points(zone_col):
    """One point per OD pair (the zone centroid); row i of origins is routed to row i of destinations."""
    pts = gpd.GeoDataFrame(demand.merge(zones, left_on=zone_col, right_on='LocationID', how='left'), crs='EPSG:4326')
    pts['id'] = range(len(pts))
    pts['id'] = pts['id'].astype(str)  # r5py prefers string ids
    return pts


origins, destinations = od_points('PULocationID'), od_points('DOLocationID')

network = r5py.TransportNetwork(osm_pbf=f'{args.raw_dir}/{args.osm_file}', gtfs=[f'{args.raw_dir}/{GTFS_NAME}.zip'])
itineraries = r5py.DetailedItineraries(
    network,
    origins=origins,
    destinations=destinations,
    departure=datetime.datetime.fromisoformat(args.departure),
    max_time=datetime.timedelta(hours=args.max_hours),
    transport_modes=[r5py.TransportMode.TRANSIT, r5py.TransportMode.WALK],
    snap_to_network=True,
    force_all_to_all=False,
)

cols = ['PULocationID', 'DOLocationID', 'option', 'segment', 'transport_mode', 'distance', 'travel_time',
        'wait_time', 'route_id', 'start_stop_id', 'end_stop_id']
itineraries = itineraries.merge(origins[['PULocationID', 'DOLocationID', 'id']], left_on='from_id', right_on='id')[cols]

os.makedirs(f'{args.out_dir}/metro', exist_ok=True)
itineraries.to_csv(f'{args.out_dir}/metro/itineraries_raw_multi.csv.gz', index=False, compression='gzip')
print(f"{len(itineraries)} segment rows, "
      f"{itineraries.groupby(['PULocationID', 'DOLocationID'])['option'].nunique().mean():.1f} options per OD")
