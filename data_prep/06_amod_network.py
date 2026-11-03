"""Manhattan driving network (OSMnx) and candidate AMoD routes -> on_demand/map.csv, on_demand/paths.csv.

The raw network is a live OpenStreetMap (Overpass) query, which changes as OSM is edited. It is saved
to --graphml on the first run and loaded from there afterwards, so keep that file with the processed
data (the date of the query is the file's modification time). Needs the OD set of metro/optimization_inputs.pkl.gz (run 04 first).
"""
import argparse
import datetime
import gzip
import os
import pickle
import networkx as nx
import osmnx as ox
import pandas as pd
from common import load_manhattan_zones, zone_centroids

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--raw_dir', default='data_nyc')
parser.add_argument('--out_dir', default='data_nyc/processed')
parser.add_argument('--graphml', default='data_nyc/manhattan_drive.graphml', help='raw OSMnx graph: loaded if present, else downloaded and saved here')
parser.add_argument('--tolerance', type=float, default=5, help='intersection consolidation tolerance (m)')
parser.add_argument('--k_paths', type=int, default=5, help='candidate routes per OD')
parser.add_argument('--penalty', type=float, default=1.2, help='length multiplier on the edges of a found route, to diversify the next one')
parser.add_argument('--default_speed', type=int, default=40, help='speed (mph) of edges without an OSM maxspeed tag; the speed limit in the tagged Manhattan edges is mostly 25')
args = parser.parse_args()

# ── Driving network: major roads, consolidated, largest strongly connected component ──
if os.path.exists(args.graphml):
    G = ox.load_graphml(args.graphml)
else:
    G = ox.graph_from_place('Manhattan, New York City, USA', network_type='drive',
                            custom_filter='["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified"]')
    ox.save_graphml(G, args.graphml)
    print(f"Queried OSM on {datetime.date.today()}, saved to {args.graphml}")
G = ox.consolidate_intersections(ox.project_graph(G), tolerance=args.tolerance, rebuild_graph=True)
G = ox.project_graph(G, to_crs='EPSG:4326')
G = G.subgraph(max(nx.strongly_connected_components(G), key=len)).copy()
assert nx.is_strongly_connected(G)
print(f"Network: {len(G.nodes):,} nodes, {len(G.edges):,} edges")

# ── Zone -> nearest network node (zone centroid) ──
zones = zone_centroids(load_manhattan_zones(args.raw_dir))
zone_node = dict(zip(zones['LocationID'], ox.nearest_nodes(G, X=zones.geometry.x.tolist(), Y=zones.geometry.y.tolist())))

with gzip.open(f'{args.out_dir}/metro/optimization_inputs.pkl.gz', 'rb') as f:
    od_pairs = sorted(pickle.load(f)['optimization_data'])


def candidate_routes(source, target):
    """{iteration: edge list}: the shortest route, then again after multiplying the length of the edges of
    the routes found so far by --penalty. Duplicate routes are dropped, so iterations (path_id suffix) may have gaps."""
    H, routes, seen = G.copy(), {}, set()
    for i in range(args.k_paths):
        try:
            path = nx.shortest_path(H, source, target, weight='length')
        except nx.NetworkXNoPath:
            break
        if len(path) < 2:  # source == target
            break
        edges = list(zip(path[:-1], path[1:]))
        if tuple(edges) not in seen:
            seen.add(tuple(edges))
            routes[i] = edges
        for u, v in edges:
            for key in H[u][v]:
                H[u][v][key]['length'] *= args.penalty
    return routes


paths = pd.DataFrame([
    {'o_zone': o, 'd_zone': d, 'path_id': f'path_{o}_{d}_{i}', 'u': u, 'v': v}
    for o, d in od_pairs
    for i, edges in candidate_routes(zone_node[o], zone_node[d]).items()
    for u, v in edges
])
n_routes = paths.groupby(['o_zone', 'd_zone'])['path_id'].nunique()
print(f"{len(n_routes)}/{len(od_pairs)} ODs with routes; routes per OD min {n_routes.min()}, max {n_routes.max()}, mean {n_routes.mean():.1f}")

# ── Edge table: length (m), lanes and speed limit (mph); OSM gaps filled with 2 lanes / --default_speed ──
edges = ox.graph_to_gdfs(G, nodes=False).reset_index()
for col in ['lanes', 'maxspeed']:
    if col not in edges:
        edges[col] = None
edges['lanes'] = edges['lanes'].apply(lambda x: int(x[0]) if isinstance(x, list) else (int(x) if pd.notnull(x) else 2))
edges['maxspeed'] = edges['maxspeed'].apply(
    lambda x: int(x[0].split()[0]) if isinstance(x, list) else (int(x.split()[0]) if isinstance(x, str) else args.default_speed))
edges = edges.rename(columns={'u': 'source', 'v': 'target'})[['source', 'target', 'length', 'highway', 'lanes', 'maxspeed']]
edges = edges.drop_duplicates(['source', 'target'])  # parallel edges: keep the first

assert set(zip(paths['u'], paths['v'])) <= set(zip(edges['source'], edges['target'])), "route uses an edge missing from the edge table"

os.makedirs(f'{args.out_dir}/on_demand', exist_ok=True)
paths.to_csv(f'{args.out_dir}/on_demand/paths.csv', index=False)
edges.to_csv(f'{args.out_dir}/on_demand/map.csv', index=False)
