import pandas as pd
import gzip
import pickle
import os
from collections import Counter
from src.constants import C_LANE
from src.transit_paths import segment_incidence

def load_nyc_network(args):
    demand_f = os.path.join(args.input_dir, args.demand_file)
    transit_f = os.path.join(args.input_dir, args.transit_file)
    map_f = os.path.join(args.input_dir, args.amod_map)
    amod_f = os.path.join(args.input_dir, args.amod_file)

    # Load data
    map_df = pd.read_csv(map_f)
    paths_df = pd.read_csv(amod_f)
    demand = pd.read_csv(demand_f)

    # Optional cap on the AMoD candidate-route set (R1:9 sensitivity). Ranks are the trailing
    # '_k' of path_id; every OD has rank 0, so the OD set is unchanged. None = keep all (default).
    cap = getattr(args, 'num_path_amod', None)
    if cap is not None:
        rank = paths_df['path_id'].str.rsplit('_', n=1).str[1].astype(int)
        paths_df = paths_df[rank < cap].reset_index(drop=True)
        print(f"AMoD candidate routes capped at {cap} per OD: {paths_df['path_id'].nunique()} paths kept")

    OD_pairs = list(paths_df.groupby(['o_zone', 'd_zone']).groups.keys())
    demand_dict = {od: demand[(demand['PULocationID'] == od[0]) & (demand['DOLocationID'] == od[1])]['trip_count'].values[0] for od in OD_pairs}

    with gzip.open(transit_f, "rb") as f:
        data = pickle.load(f)

    # Candidate transit paths per OD. The old pickle stores a single dict per OD, the
    # multi-path one a list sorted by travel time with index 0 = the old reference path.
    pt_paths = {}
    for od in OD_pairs:
        paths = data["optimization_data"][od]
        if isinstance(paths, dict):
            paths = [paths]
        pt_paths[od] = paths[:args.num_path]

    counts = Counter(len(p) for p in pt_paths.values())
    if args.num_path > 1 and set(counts) == {1}:
        raise ValueError(
            f"--num_path={args.num_path} but the transit file {transit_f} has a single path per OD. "
            "Pass --transit_file metro/optimization_inputs_k5.pkl.gz."
        )
    print(f"Transit candidate paths per OD: {dict(sorted(counts.items()))}")

    lines_dict = data["lines"]

    # Calculate Network Constants
    L = list(lines_dict.keys())
    E_a = list(zip(map_df['source'], map_df['target']))
    road_nodes = list(set(map_df['source']).union(set(map_df['target'])))

    # Free-flow travel time and capacity
    map_df['T0'] = map_df['length'] / (map_df['maxspeed'] * 0.44704 * 60)
    map_df['Capacity'] = map_df['lanes'] * C_LANE
    dist_map = map_df.set_index(['source', 'target'])['length'].to_dict()
    T0_map = map_df.set_index(['source', 'target'])['T0'].to_dict()
    Cap_map = map_df.set_index(['source', 'target'])['Capacity'].to_dict()

    # Background Traffic
    b_map = {edge: Cap_map[edge] * args.bl for edge in E_a}

    baseline_road_cost = sum(
        b_map[e] * T0_map[e] * (1 + args.alpha * (b_map[e] / Cap_map[e])**args.beta) 
        for e in E_a
    )

    # Zone to Node mapping
    zone_to_node = {}
    for (o, d), group in paths_df.groupby(['o_zone', 'd_zone']):
        path0 = group[group['path_id'].str.endswith('_0')]
        if not path0.empty:
            zone_to_node[o] = path0.iloc[0]['u']
            zone_to_node[d] = path0.iloc[-1]['v']

    # Pack everything into a clean dictionary
    net_data = {
        'map_df': map_df,
        'paths_df': paths_df,
        'OD_pairs': OD_pairs,
        'pt_paths': pt_paths,
        'seg_to_odp': segment_incidence(pt_paths),
        'lines_dict': lines_dict,
        'L': L,
        'E_a': E_a,
        'road_nodes': road_nodes,
        'dist_map': dist_map,
        'T0_map': T0_map,
        'Cap_map': Cap_map,
        'b_map': b_map,
        'baseline_road_cost': baseline_road_cost,
        'zone_to_node': zone_to_node
    }

    return net_data, demand_dict