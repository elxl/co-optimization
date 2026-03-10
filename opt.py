import gurobipy as gp
from gurobipy import GRB
import pickle
import pandas as pd
import numpy as np
import gzip, pickle,os
import argparse
from pricing import recover_amod_price

parser = argparse.ArgumentParser(description="Co optimization")
parser.add_argument(
    '-o', "--output_dir", 
    type=str,
    default='./results',
    help='Directory for result files'
)
parser.add_argument(
    "--input_dir", 
    type=str,
    default='./data_nyc/processed',
    help='Directory for input files'
)
parser.add_argument(
    "--demand_file", 
    type=str,
    default='demand.csv',
    help='Directory for demand file'    
)
parser.add_argument(
    "--transit_file", 
    type=str,
    default='metro/optimization_inputs.pkl.gz',
    help='Directory for transit file'    
)
parser.add_argument(
    "--amod_file", 
    type=str,
    default='on_demand/paths.csv',
    help='Directory for amod file'    
)
parser.add_argument(
    "--amod_map", 
    type=str,
    default='on_demand/map.csv',
    help='Directory for driving network'    
)
parser.add_argument(
    "--bl", 
    type=float,
    default=1.0,
    help='Background traffic level'
)
parser.add_argument(
    "--ca", 
    type=float,
    default=1.0,
    help='AMoD operation cost per km'
)
parser.add_argument(
    "--cl", 
    type=float,
    default=100,
    help='Metro operation cost per line per hour'
)
parser.add_argument(
    "--walk", 
    type=float,
    default=3.6,
    help='Walking speed (mph)'
)
parser.add_argument(
    "--vot", 
    type=float,
    default=0.4,
    help='Environmental price for driving'
)
parser.add_argument(
    "--vot1", 
    type=float,
    default=0.3,
    help='Value of time for passengers'
)
parser.add_argument(
    "--mip", 
    type=float,
    default=0.01,
    help='Optimization gap.'
)
parser.add_argument(
    "--alpha", 
    type=float,
    default=0.15,
    help='alpha parameter of BPR.'
)
parser.add_argument(
    "--beta", 
    type=float,
    default=4,
    help='beta parameter of BRP.'
)
parser.add_argument(
    "--comp", 
    type=bool,
    default=False,
    help='Save result file to compact format (parquet).'
)
parser.add_argument(
    "--reg", 
    type=int,
    default=0,
    help='Regularization. (0: no regularization, 1: quadratic, 2: entropy)'
)
parser.add_argument(
    "--theta", 
    type=float,
    default=0.2,
    help='Coefficient for regularization term (higher theta = more spread out; lower theta = closer to original objective)'
)

args = parser.parse_args()
demand_f = os.path.join(args.input_dir, args.demand_file)
transit_f = os.path.join(args.input_dir, args.transit_file)
map_f = os.path.join(args.input_dir, args.amod_map)
amod_f = os.path.join(args.input_dir, args.amod_file)
output_dir = args.output_dir
reg = args.reg
reg_dict = {0:"", 1:"_quad", 2:"_ent"}

#--------Data Preparation-----------#
# Load on-demand map and routes
map_df = pd.read_csv(map_f) # Road edges (source, target, length, lanes, maxspeed)
paths_df = pd.read_csv(amod_f) # AMOD candidate paths (o_zone, d_zone, path_id, u, v)

# Load demand data
demand = pd.read_csv(demand_f)

OD_pairs = list(paths_df.groupby(['o_zone', 'd_zone']).groups.keys())
demand_dict = {od:demand[(demand['PULocationID']==od[0]) & (demand['DOLocationID']==od[1])]['trip_count'].values[0] for od in OD_pairs}

# Load transit lines and routes data
with gzip.open(transit_f, "rb") as f:
    data = pickle.load(f)

opt_data = data["optimization_data"] # {(O, D): {walk_dist, tau_pt, gamma_odl, eta_e_od}}
opt_data = {(o, d): info for (o, d), info in opt_data.items() if (o, d) in OD_pairs} # Filter to only OD pairs with amod paths
lines_dict = data["lines"] # {line_id: [(u,v), ...]}

# Start with samll examples to debug
# sub_od = list(opt_data.keys())[:10]
# sub_od_set = set(map(tuple, sub_od))  # ensure (o, d) tuple format
# opt_data = {k: v for k, v in opt_data.items() if k in sub_od_set}
# paths_df = paths_df[
#     paths_df.apply(lambda r: (r['o_zone'], r['d_zone']) in sub_od_set, axis=1)
# ]
# demand_dict = {k: v for k, v in demand_dict.items() if k in sub_od_set}
# OD_pairs = list(opt_data.keys())


#------Optimization Setup-------#
# Parameters and sets
VOT_ENV = args.vot            # Environmental effect of congestion ($/min)
VOT_USER = args.vot1      # Value of time for passengers
C_ij = args.ca           # AMOD unit operating cost ($/km)
K_l = args.cl        # Transit unit operating cost ($/hr per line)
C_l_unit = 1400       # Passengers per train capacity
alpha = args.alpha         # Standard BPR alpha
beta = args.beta          # Standard BPR beta
C_lane = 700        # Capacity per lane (vehicles/hr)
S_set = [2, 3, 4, 5, 6, 12, 20]   # Frequency candidates (trains per hour)
W_SPEED = args.walk * 1000/60 # Walking speed m/minute
AMOD_WAIT = 5 # AMoD average pickup time
BL = args.bl # Background traffic level
base_load_factor = 0.6
PT = 3.0 # Fixed transit fare

if reg != 0:
    suffix = f"_{BL}_{C_ij}_{K_l}{reg_dict[reg]}_theta{args.theta}"
else:
    suffix = f"_{BL}_{C_ij}_{K_l}"

L = list(lines_dict.keys()) # Lines
E_a = list(zip(map_df['source'], map_df['target'])) # Edges
road_nodes = list(set(map_df['source']).union(set(map_df['target']))) # Nodes

# Free-flow travel time and capacity
map_df['T0'] = map_df['length'] / (map_df['maxspeed'] * 0.44704 * 60) # minute
map_df['Capacity'] = map_df['lanes'] * C_lane
dist_map = map_df.set_index(['source', 'target'])['length'].to_dict()
T0_map = map_df.set_index(['source', 'target'])['T0'].to_dict()
Cap_map = map_df.set_index(['source', 'target'])['Capacity'].to_dict()

# Background Traffic
b_map = {edge: Cap_map[edge] * BL for edge in E_a}

# Calculate Baseline Cost (Constant to be subtracted)
# Baseline = sum( b * T0 * (1 + alpha * (b/C)^beta) )
baseline_road_cost = sum(
    b_map[e] * T0_map[e] * (1 + alpha * (b_map[e] / Cap_map[e])**beta) 
    for e in E_a
)

# Identify zone id to node id
zone_to_node = {}
for (o, d), group in paths_df.groupby(['o_zone', 'd_zone']):
    # For each OD, the first 'u' of path_0 is the origin node
    # The last 'v' of path_0 is the destination node
    path0 = group[group['path_id'].str.endswith('_0')]
    if not path0.empty:
        zone_to_node[o] = path0.iloc[0]['u']
        zone_to_node[d] = path0.iloc[-1]['v']


#------Build Model-------#
m = gp.Model("Multimodal_BPR_Optimization")
if reg != 2:
    m.params.NonConvex = 0 # Allow non-convex quadratic/non-linear terms
else:
    m.params.NonConvex = 1
m.params.MIPGap = args.mip

# Variables
x_ap = m.addVars(paths_df['path_id'].unique(), lb=0, name="x_path") # Flow on AMoD path
x_a = m.addVars(OD_pairs, lb=1e-6, ub=[demand_dict[od] for od in OD_pairs], name="x_a")
r = m.addVars(E_a, lb=0, name="r_ij") # Rebalancing flow
f = m.addVars(E_a, lb=0, name="f_ij") # AMOD flow
v = m.addVars(E_a, lb=[b_map[e] for e in E_a], name="v_total") # Total volume (b + f)
v_ratio = m.addVars(E_a, lb=0, name="v_ratio")
v_ratio_pow = m.addVars(E_a, lb=0, name="v_ratio_pow") # Auxiliary for v^(beta+1) in BPR
x_pt = m.addVars(OD_pairs, lb=1e-6, ub=[demand_dict[od] for od in OD_pairs], name="x_pt") # Flow on transit route

z = m.addVars(L, S_set, vtype=GRB.BINARY, name="z_ls") # z_ls=1 if line l has frequency s
s_l = m.addVars(L, lb=min(S_set), name="s_l") # Frequency of line l TODO: discrete/continuous?
# Sparse Indexing for w (auxilary variiable for linearization)
# Create a list of valid (OD, Line, S) triplets
w_indices = []
for od, info in opt_data.items():
    for line in info['gamma_odl']:
        for s in S_set:
            w_indices.append((od, line, s))

# Create w variables ONLY for relevant combinations
w = m.addVars(w_indices, lb=0, ub=[demand_dict[od] for od, l, s in w_indices], name="w_sparse")

# Constraints
# 1. Flow on road
for e in Cap_map.keys():
    p_on_e = paths_df[(paths_df['u'] == e[0]) & (paths_df['v'] == e[1])]['path_id'].unique() # Find all paths contain edge e
    m.addConstr(f[e] == r[e] + x_ap.sum(p_on_e))
    # Total flow on edge and BPR
    m.addConstr(v[e] == b_map[e] + f[e])
    # v_ratio will be around 0.5 to 1.5
    m.addConstr(v_ratio[e] == v[e] / Cap_map[e])
    # (v/C)^5 will be around 0.03 to 7.6
    m.addGenConstrPow(v_ratio[e], v_ratio_pow[e], beta+1)

# 2. FLOW CONSERVATION (Rebalancing)
# Net change in vehicles at node v: Inflow - Outflow = Net Demand
for node in road_nodes:
    # Vehicles arriving/departing node by rebalancing
    rebal_in = r.sum('*', node)
    rebal_out = r.sum(node, '*')
    
    # Vehicles consumed by departing passengers at this node
    pass_out = gp.quicksum(x_a[o, d] for o, d in OD_pairs if zone_to_node.get(o) == node)
    # Vehicles produced by arriving passengers at this node
    pass_in = gp.quicksum(x_a[o, d] for o, d in OD_pairs if zone_to_node.get(d) == node)
    
    # Equation: Vehicles In + Pass In = Vehicles Out + Pass Out
    m.addConstr(rebal_in + pass_in == rebal_out + pass_out, name=f"flow_bal_{node}")

# 3. Demand split and Path mapping
for od in opt_data.keys():
    m.addConstr(x_a[od] + x_pt[od] == demand_dict[od])
    rel_paths = paths_df[(paths_df['o_zone'] == od[0]) & (paths_df['d_zone'] == od[1])]['path_id'].unique()
    m.addConstr(x_a[od] == x_ap.sum(rel_paths))

# 4. Transit Capacity & Frequency selection
for l in L:
    m.addConstr(z.sum(l, '*') == 1) # Frequency selection
    m.addConstr(s_l[l] == gp.quicksum(s * z[l, s] for s in S_set))
    for u_s, v_s in lines_dict[l]:
        # Capacity logic: passengers whose path includes this directed line segment
        pass_on_seg = gp.quicksum(x_pt[od] for od in opt_data.keys() 
                                  if (u_s, v_s, l) in opt_data[od]['eta_e_od'])
        m.addConstr(pass_on_seg <= (1-base_load_factor) * C_l_unit * s_l[l])

# 5. Linearization
for (od, l, s) in w.keys():
    m.addConstr(w[od, l, s] <= x_pt[od])
    m.addConstr(w[od, l, s] <= demand_dict[od] * z[l, s]) # Big-M = max demand
    m.addConstr(w[od, l, s] >= x_pt[od] - demand_dict[od] * (1 - z[l, s]))

# Objective
road_time_cost = gp.quicksum(T0_map[e] * (v[e] + alpha * Cap_map[e] * v_ratio_pow[e]) for e in E_a) - baseline_road_cost
amod_wait = gp.quicksum(AMOD_WAIT * x_a[od] for od in OD_pairs)
transit_time_fixed = gp.quicksum(x_pt[od] * opt_data[od]['tau_pt'] for od in OD_pairs) # transit in vehicle
transit_walk_fixed = gp.quicksum(x_pt[od] * opt_data[od]['walking_dist']/W_SPEED for od in OD_pairs) # transit walking
transit_waiting = 0.5 * 60 * gp.quicksum((1/s) * w[od, l, s] for (od, l, s) in w.keys()) # transit waiting (pay attention to the unit of time)
amod_op_cost = gp.quicksum(C_ij * f[u, v] * (dist_map[u, v] / 1000.0) for u, v in E_a)
op_cost = gp.quicksum(K_l * s_l[l] for l in L)
# obj = VOT * (road_time_cost + transit_time_fixed) + VOT_OTHER * (transit_waiting + transit_walk_fixed + amod_wait)+ amod_op_cost + op_cost
obj = VOT_ENV * road_time_cost  + VOT_USER * (transit_time_fixed + transit_waiting + transit_walk_fixed + amod_wait)+ amod_op_cost + op_cost

# --- Define theta (the "smoothing" parameter) ---
# Larger theta = more spread out (50/50); Smaller theta = back to 100/0.
if reg == 0:
    m.setObjective(obj, GRB.MINIMIZE)
elif reg == 1:
    m.setObjective(obj + (1.0/args.theta) * gp.quicksum(x_a[od]**2 + x_pt[od]**2 for od in OD_pairs), GRB.MINIMIZE)
elif reg == 2:
    # Larger theta = more spread out (50/50); Smaller theta = back to 100/0.
    entropy_terms = []
    n_points = 30

    for od in OD_pairs:
        # Create auxiliary variables for the entropy values: f(x) = x * log(x)
        # The minimum value of x*log(x) is -1/e (~ -0.368), so lb=-1 is safe.
        ent_a = m.addVar(lb=-1.0, name=f"ent_a_{od}")
        ent_pt = m.addVar(lb=-1.0, name=f"ent_pt_{od}")

        # Generate coordinates for the PWL approximation
        # Range is from epsilon to the maximum possible demand for this OD
        max_d = max(1.0, demand_dict[od]) 
        x_pts = np.geomspace(1e-6, max_d, num=n_points).tolist()
        y_pts = [p * np.log(p) for p in x_pts]
        
        # Define the PWL constraints (This is linear/MIP compatible)
        m.addGenConstrPWL(x_a[od], ent_a, x_pts, y_pts)
        m.addGenConstrPWL(x_pt[od], ent_pt, x_pts, y_pts)
        
        entropy_terms.append(ent_a + ent_pt)

    # Update Objective: Minimize (Social Costs + 1/theta * Entropy)
    m.setObjective(obj + (1.0/args.theta) * gp.quicksum(entropy_terms), GRB.MINIMIZE)

# m.setObjective(VOT * (road_time_cost + transit_time_fixed) + VOT_OTHER * (transit_waiting + transit_walk_fixed + amod_wait)+ amod_op_cost + op_cost, GRB.MINIMIZE)

m.optimize()

if m.status == GRB.INFEASIBLE:
    print("Model is infeasible. Finding the IIS...")
    m.computeIIS()
    m.write("model.ilp")
    print("IIS written to file 'model.ilp'. Open this file in a text editor to see the conflict.")
    
    # Optional: Print the constraints in the IIS directly to the console
    for c in m.getConstrs():
        if c.IISConstr:
            print(f"Infeasible Constraint: {c.ConstrName}")


#-------------Summarize results----------------#
if m.status == GRB.OPTIMAL:
    # 1. Frquency setting, usage and operation cost
    transit_summary = []

    for l in L:
        # Sum transit flow for all OD pairs that use this line l
        # We check if l is in the 'gamma_odl' list for that OD pair
        line_passengers = sum(x_pt[od].X for od in OD_pairs if l in opt_data[od]['gamma_odl'])
        
        transit_summary.append({
            'Line_ID': l,
            'Frequency_vph': round(s_l[l].X, 2),
            'Total_Passengers': round(line_passengers, 2),
            'Load_Factor': base_load_factor + round(line_passengers / (C_l_unit * s_l[l].X), 3) if s_l[l].X > 0 else base_load_factor
        })

    # Create DataFrame and save to CSV
    df_transit = pd.DataFrame(transit_summary)
    if args.comp:
        df_transit.to_parquet(os.path.join(output_dir, f"transit_line_usage{suffix}.parquet"))
    else:
        df_transit.to_csv(os.path.join(output_dir, f"transit_line_usage{suffix}.csv"),index=False)

    total_transit_op_cost = sum(K_l * s_l[l].X for l in L)
    print(f"Transit Operation Cost: ${total_transit_op_cost:,.2f} (K_l = {K_l})")

    # 2. AMoD total distance
    total_rebal = sum(r[e].X * dist_map[e] for e in E_a) / 1000
    total_pass_dist = sum(x_ap[p].X * sum(dist_map[row.u, row.v] for row in paths_df[paths_df['path_id']==p].itertuples()) for p in x_ap.keys() if x_ap[p].X > 0) / 1000
    print(f"\nTotal AMoD Pax distance: {total_pass_dist:.2f} km")
    print(f"Total Rebalancing distance: {total_rebal:.2f} km")

    # 3. Mode split and service quality
    # Calculate Congested Time for every edge based on total volume (v)
    congested_time_map = {}
    for e in E_a:
        v_total = v[e].X
        ratio = v_total / Cap_map[e]
        # BPR: T = T0 * (1 + alpha * (ratio^beta))
        congested_time_map[e] = T0_map[e] * (1 + alpha * (ratio**beta))
    
    # Calculate Line Wait Times (0.5 * 1/frequency * 60 min)
    line_wait_map = {l: (0.5 * 60 / s_l[l].X) if s_l[l].X > 0 else 0 for l in L}

    detailed_summary = []
    for od in OD_pairs:
        # --- AMoD Metrics & Path Usage ---
        rel_paths = paths_df[(paths_df['o_zone'] == od[0]) & (paths_df['d_zone'] == od[1])]['path_id'].unique()
        amod_flow = x_a[od].X
        
        # Calculate how many paths are actually carrying flow
        used_paths_count = sum(1 for p in rel_paths if x_ap[p].X > 1e-6)
        
        # Potential metrics logic
        best_p = sorted(rel_paths)[0] if len(rel_paths) > 0 else None
        
        if amod_flow > 1e-6:
            # Weighted average of the paths actually used
            pot_amod_dist = sum(x_ap[p].X * sum(dist_map[r.u, r.v] for r in paths_df[paths_df['path_id']==p].itertuples()) 
                               for p in rel_paths) / (amod_flow * 1000.0)
            pot_amod_time = sum(x_ap[p].X * (sum(congested_time_map[r.u, r.v] for r in paths_df[paths_df['path_id']==p].itertuples())+AMOD_WAIT) 
                               for p in rel_paths) / amod_flow
            # Note: You might want to add AMOD_WAIT here as well for consistency
        elif best_p:
            p_edges = paths_df[paths_df['path_id'] == best_p]
            pot_amod_dist = sum(dist_map[r.u, r.v] for r in p_edges.itertuples()) / 1000.0
            pot_amod_time = sum(congested_time_map[r.u, r.v] for r in p_edges.itertuples()) + AMOD_WAIT
        else:
            pot_amod_dist, pot_amod_time = 0, 0

        # --- Transit Potential Metrics ---
        pot_transit_wait = sum(line_wait_map[l] for l in opt_data[od]['gamma_odl'])
        walking_time = opt_data[od]['walking_dist'] / W_SPEED
        pot_transit_total_time = opt_data[od]['tau_pt'] + walking_time + pot_transit_wait

        detailed_summary.append({
            'O': od[0], 'D': od[1],
            'Demand': demand_dict[od],
            'AMoD_Flow': round(amod_flow, 2),
            'Transit_Flow': round(x_pt[od].X, 2),
            'AMoD_Paths_Used': used_paths_count,
            'Pot_AMoD_Time_min': round(pot_amod_time, 2),
            'Pot_AMoD_Dist_km': round(pot_amod_dist, 2),
            'Pot_Transit_Wait_min': round(pot_transit_wait, 2),
            'Pot_Transit_Total_min': round(pot_transit_total_time, 2)
        })

    df_results = pd.DataFrame(detailed_summary)
    # Calculate AMoD price
    df_results = recover_amod_price(
        df_results,
        VOT_USER,
        PT
    )

    if args.comp:
        df_results.to_parquet(os.path.join(output_dir, f"mode_split{suffix}.parquet"))
    else:
        df_results.to_csv(os.path.join(output_dir, f"mode_split{suffix}.csv"),index=False)

    # Calculate total flows
    total_amod = df_results['AMoD_Flow'].sum()
    total_transit = df_results['Transit_Flow'].sum()
    total_demand = df_results['Demand'].sum()

    # Calculate shares as percentages
    amod_share = (total_amod / total_demand) * 100
    transit_share = (total_transit / total_demand) * 100

    print(f"Total Demand: {total_demand:,.0f} passengers")
    print(f"Total AMoD Flow: {total_amod:,.2f} ({amod_share:.2f}%)")
    print(f"Total Transit Flow: {total_transit:,.2f} ({transit_share:.2f}%)")

    # 4. Background traffic impact
    total_bg_increase_min = 0
    
    for e in E_a:
        # 1. Congested Travel Time (with AMoD)
        # T = T0 * (1 + alpha * (v_total / Capacity)^beta)
        v_final = v[e].X
        t_congested = T0_map[e] * (1 + alpha * ( (v_final / Cap_map[e]) ** beta ))
        
        # 2. Baseline Travel Time (Background traffic only)
        # T_base = T0 * (1 + alpha * (b_map / Capacity)^beta)
        t_baseline = T0_map[e] * (1 + alpha * ( (b_map[e] / Cap_map[e]) ** beta ))
        
        # 3. Total increase for all background vehicles on this edge
        # Increase = Volume_bg * (T_congested - T_baseline)
        edge_increase = b_map[e] * (t_congested - t_baseline)
        total_bg_increase_min += edge_increase

    print("\n--- Background Traffic Impact ---")
    print(f"Total travel time increase for background traffic: {total_bg_increase_min:,.2f} minutes")

    # 5. AMoD flow on edge
    edge_results = []
        
    for e in E_a:
        amod_flow = f[e].X
        total_vol = v[e].X
        capacity = Cap_map[e]
        background = b_map[e]
        
        # Calculate the travel times for verification
        # T = T0 * (1 + alpha * (v/C)**beta)
        t_baseline = T0_map[e] * (1 + alpha * (background / capacity)**beta)
        t_congested = T0_map[e] * (1 + alpha * (total_vol / capacity)**beta)
        time_diff = t_congested - t_baseline
        
        edge_results.append({
            'source': e[0],
            'target': e[1],
            'amod_flow_f_ij': round(amod_flow, 4),
            'background_vol': round(background, 2),
            'total_vol_v': round(total_vol, 2),
            'capacity': capacity,
            'v_c_ratio': round(total_vol / capacity, 4),
            'time_increase_min': round(time_diff, 6)
        })

    # Create DataFrame and save
    df_edges = pd.DataFrame(edge_results)
    
    # Filter for edges that actually have AMoD flow to see where the fleet is
    df_active = df_edges[df_edges['amod_flow_f_ij'] > 1e-4]
    
    print(f"\n--- Edge Flow Analysis ---")
    print(f"Total edges with AMoD flow: {len(df_active)} / {len(df_edges)}")
    
    # if not df_active.empty:
    #     print("\nTop 10 Edges by AMoD Flow:")
    #     print(df_active.sort_values(by='amod_flow_f_ij', ascending=False).head(10))
    
    # Save the full network state
    if args.comp:
        df_edges.to_parquet(os.path.join(output_dir, f"edge_flows_full{suffix}.parquet"))
    else:
        df_edges.to_csv(os.path.join(output_dir, f"edge_flows_full{suffix}.csv"),index=False)

    # Final summary
    avg_freq = df_transit['Frequency_vph'].mean()
    avg_wait = df_results['Pot_Transit_Wait_min'].mean()
    total_amod_op_cost = sum(C_ij * f[e].X * (dist_map[e] / 1000.0) for e in E_a)
    total_physical_cost = (
            VOT_ENV * road_time_cost.getValue() + 
            VOT_USER * (transit_time_fixed.getValue() + transit_waiting.getValue() + 
                        transit_walk_fixed.getValue() + amod_wait.getValue()) + 
            amod_op_cost.getValue() + 
            op_cost.getValue()
        )
    run_summary = {
        # Metadata / Inputs
        "BL": BL,
        "K_l": K_l,
        "C_ij": C_ij,
        "Total_Social_Cost": total_physical_cost,
        
        # Mode Split
        "Total_Demand": total_demand,
        "AMoD_Flow": total_amod,
        "Transit_Flow": total_transit,
        "AMoD_Share_pct": round(amod_share, 2),
        "Transit_Share_pct": round(transit_share, 2),

        # Supply Side Metrics
        "Avg_Transit_Freq": round(avg_freq, 2),
        "Avg_Wait_Time_min": round(avg_wait, 2),
        "Transit_Op_Cost": round(total_transit_op_cost, 2),
        "AMoD_Op_Cost": round(total_amod_op_cost, 2),
        
        # Road/Congestion Metrics
        "Rebalancing_Dist_km": round(total_rebal, 2),
        "AMoD_Pax_Dist_km": round(total_pass_dist, 2),
        "Rebal_Ratio": round(total_rebal / total_pass_dist, 3) if total_pass_dist > 0 else 0,
        "BG_Delay_Total_min": round(total_bg_increase_min, 2),
    }
    df_summary = pd.DataFrame([run_summary])
    df_summary.to_csv(os.path.join(output_dir, f"summary{suffix}.csv"), index=False)