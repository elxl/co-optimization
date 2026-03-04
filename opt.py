import gurobipy as gp
from gurobipy import GRB
import pickle
import pandas as pd
import numpy as np
import gzip, pickle


#--------Data Preparation-----------#
# Load transit lines and routes data
with gzip.open("data_nyc/processed/metro/optimization_inputs.pkl.gz", "rb") as f:
    data = pickle.load(f)

opt_data = data["optimization_data"] # {(O, D): {walk_dist, tau_pt, gamma_odl, eta_e_od}}
lines_dict = data["lines"] # {line_id: [(u,v), ...]}

# Load on-demand map and routes
map_df = pd.read_csv("data_nyc/processed/on_demand/map.csv") # Road edges (source, target, length, lanes, maxspeed)
paths_df = pd.read_csv("data_nyc/processed/on_demand/paths.csv") # AMOD candidate paths (o_zone, d_zone, path_id, u, v)

# Load demand data
demand = pd.read_csv("data_nyc/processed/demand.csv")

OD_pairs = list(opt_data.keys())
demand_dict = {od:demand[(demand['PULocationID']==od[0]) & (demand['DOLocationID']==od[1])]['trip_count'].values[0] for od in OD_pairs}

# Start with samll examples to debug
sub_od = list(opt_data.keys())[:10]
sub_od_set = set(map(tuple, sub_od))  # ensure (o, d) tuple format
opt_data = {k: v for k, v in opt_data.items() if k in sub_od_set}
paths_df = paths_df[
    paths_df.apply(lambda r: (r['o_zone'], r['d_zone']) in sub_od_set, axis=1)
]
demand_dict = {k: v for k, v in demand_dict.items() if k in sub_od_set}
OD_pairs = list(opt_data.keys())


#------Optiization Setup-------#
# Parameters and sets
VOT = 0.3            # Value of time ($/min)
VOT_OTHER = 0.33      # Value of time (walk/stand)
C_ij = 0.8           # AMOD unit operating cost ($/km)
K_l = 200         # Transit unit operating cost ($/hr per line)
C_l_unit = 1400       # Passengers per train capacity
alpha = 0.15         # Standard BPR alpha
beta = 4.0           # Standard BPR beta
C_lane = 1800        # Capacity per lane (vehicles/hr)
S_set = [4, 6, 12, 20]   # Frequency candidates (trains per hour)
W_SPEED = 3.6 * 1000/60 # Walking speed m/minute
AMOD_WAIT = 3 # TODO: add AMoD waiting time

L = list(lines_dict.keys()) # Lines
E_a = list(zip(map_df['source'], map_df['target'])) # Edges
road_nodes = list(set(map_df['source']).union(set(map_df['target']))) # Nodes

# Free-flow travel time and capacity
map_df['T0'] = map_df['length'] / (map_df['maxspeed'] * 0.44704 * 60) # minute
map_df['Capacity'] = map_df['lanes'] * C_lane
dist_map = map_df.set_index(['source', 'target'])['length'].to_dict()
T0_map = map_df.set_index(['source', 'target'])['T0'].to_dict()
Cap_map = map_df.set_index(['source', 'target'])['Capacity'].to_dict()

# Background Traffic (e.g., 20% of capacity)
b_map = {edge: Cap_map[edge] * 0.5 for edge in E_a}

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
m.params.NonConvex = 0 # Allow non-convex quadratic/non-linear terms

# Variables
x_ap = m.addVars(paths_df['path_id'].unique(), lb=0, name="x_path") # Flow on AMoD path
x_a = m.addVars(OD_pairs, lb=0, ub=[demand_dict[od] for od in OD_pairs], name="x_a")
r = m.addVars(E_a, lb=0, name="r_ij") # Rebalancing flow
f = m.addVars(E_a, lb=0, name="f_ij") # AMOD flow
v = m.addVars(E_a, lb=[b_map[e] for e in E_a], name="v_total") # Total volume (b + f)
v_ratio = m.addVars(E_a, lb=0, name="v_ratio")
v_ratio_pow = m.addVars(E_a, lb=0, name="v_ratio_pow") # Auxiliary for v^(beta+1) in BPR
x_pt = m.addVars(OD_pairs, lb=0, ub=[demand_dict[od] for od in OD_pairs], name="x_pt") # Flow on transit route

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
        m.addConstr(pass_on_seg <= C_l_unit * s_l[l])

# 5. Linearization
for (od, l, s) in w.keys():
    m.addConstr(w[od, l, s] <= x_pt[od])
    m.addConstr(w[od, l, s] <= demand_dict[od] * z[l, s]) # Big-M = max demand
    m.addConstr(w[od, l, s] >= x_pt[od] - demand_dict[od] * (1 - z[l, s]))

# Objective
road_time_cost = gp.quicksum(T0_map[e] * (v[e] + alpha * Cap_map[e] * v_ratio_pow[e]) for e in E_a) - baseline_road_cost
transit_time_fixed = gp.quicksum(x_pt[od] * opt_data[od]['tau_pt'] for od in OD_pairs)
transit_walk_fixed = gp.quicksum(x_pt[od] * opt_data[od]['walking_dist']/W_SPEED for od in OD_pairs)
transit_waiting = 0.5 * gp.quicksum((1/s) * w[od, l, s] for (od, l, s) in w.keys())
amod_op_cost = gp.quicksum(C_ij * f[u, v] * (dist_map[u, v] / 1000.0) for u, v in E_a)
op_cost = gp.quicksum(K_l * s_l[l] for l in L)

m.setObjective(VOT * (road_time_cost + transit_time_fixed) + VOT_OTHER * (transit_waiting + transit_walk_fixed)+ amod_op_cost + op_cost, GRB.MINIMIZE)

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
    # 1. Frquency setting
    freq_data = []
    for l in L:
        for s in S_set:
            if z[l,s].X > 0.5:
                freq_data.append({'Line': l, 'Selected Frequency': s})
    print(pd.DataFrame(freq_data).head(10))

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
        # BPR: T = T0 * (1 + alpha * (v/C)^beta)
        congested_time_map[e] = T0_map[e] * (1 + alpha * (ratio**beta))
    
    # Calculate Line Wait Times (0.5 * 1/frequency)
    # Even if no one uses a line, it has a frequency decided by the model
    line_wait_map = {l: (0.5 * 60 / s_l[l].X) if s_l[l].X > 0 else 0 for l in L}

    detailed_summary = []
    for od in OD_pairs:
        # --- AMoD Potential Metrics ---
        rel_paths = paths_df[(paths_df['o_zone'] == od[0]) & (paths_df['d_zone'] == od[1])]['path_id'].unique()
        amod_flow = x_a[od].X
        
        # We calculate the metrics for the best candidate path (path_0) 
        # as the 'potential' if no flow is assigned.
        best_p = sorted(rel_paths)[0] if len(rel_paths) > 0 else None
        
        if amod_flow > 1e-6:
            # Weighted average of the paths actually used
            pot_amod_dist = sum(x_ap[p].X * sum(dist_map[r.u, r.v] for r in paths_df[paths_df['path_id']==p].itertuples()) 
                               for p in rel_paths) / (amod_flow * 1000.0)
            pot_amod_time = sum(x_ap[p].X * sum(congested_time_map[r.u, r.v] for r in paths_df[paths_df['path_id']==p].itertuples()) 
                               for p in rel_paths) / amod_flow
        elif best_p:
            # Potential metrics based on the shortest candidate path
            p_edges = paths_df[paths_df['path_id'] == best_p]
            pot_amod_dist = sum(dist_map[r.u, r.v] for r in p_edges.itertuples()) / 1000.0
            pot_amod_time = sum(congested_time_map[r.u, r.v] for r in p_edges.itertuples())
        else:
            pot_amod_dist, pot_amod_time = 0, 0

        # --- Transit Potential Metrics ---
        # Wait time is the sum of wait times for all lines in the passenger's route
        pot_transit_wait = sum(line_wait_map[l] for l in opt_data[od]['gamma_odl'])
        walking_time = opt_data[od]['walking_dist'] / W_SPEED
        pot_transit_total_time = opt_data[od]['tau_pt'] + walking_time + pot_transit_wait

        detailed_summary.append({
            'O': od[0], 'D': od[1],
            'Demand': demand_dict[od],
            'AMoD_Flow': round(amod_flow, 2),
            'Transit_Flow': round(x_pt[od].X, 2),
            'Pot_AMoD_Time_min': round(pot_amod_time, 2),
            'Pot_AMoD_Dist_km': round(pot_amod_dist, 2),
            'Pot_Transit_Wait_min': round(pot_transit_wait, 2),
            'Pot_Transit_Total_min': round(pot_transit_total_time, 2)
        })
    df_results = pd.DataFrame(detailed_summary)
    print("\n--- Detailed Mode Potential Metrics ---")
    print(df_results.sort_values(by='Demand', ascending=False))

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

    # Convert to monetary value using VOT if needed
    total_bg_cost_increase = total_bg_increase_min * VOT

    print("\n--- Background Traffic Impact ---")
    print(f"Total travel time increase for background traffic: {total_bg_increase_min:,.2f} minutes")
    print(f"Total external congestion cost: ${total_bg_cost_increase:,.2f}")