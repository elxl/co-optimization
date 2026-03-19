import pandas as pd
import gzip
import pickle
import os
from src.constants import C_LANE

def load_nyc_network(args):
    demand_f = os.path.join(args.input_dir, args.demand_file)
    transit_f = os.path.join(args.input_dir, args.transit_file)
    map_f = os.path.join(args.input_dir, args.amod_map)
    amod_f = os.path.join(args.input_dir, args.amod_file)

    # Load data
    map_df = pd.read_csv(map_f)
    paths_df = pd.read_csv(amod_f)
    demand = pd.read_csv(demand_f)

    OD_pairs = list(paths_df.groupby(['o_zone', 'd_zone']).groups.keys())
    demand_dict = {od: demand[(demand['PULocationID'] == od[0]) & (demand['DOLocationID'] == od[1])]['trip_count'].values[0] for od in OD_pairs}

    with gzip.open(transit_f, "rb") as f:
        data = pickle.load(f)

    opt_data = data["optimization_data"]
    opt_data = {(o, d): info for (o, d), info in opt_data.items() if (o, d) in OD_pairs}
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
        'opt_data': opt_data,
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