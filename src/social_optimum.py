import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
from pricing import recover_amod_price
from src.constants import S_SET, AMOD_WAIT, BASE_LOAD_FACTOR, MPH_TO_M_PER_MIN, TRANSIT_FARE
from src.transit_paths import choose_path, path_time_bounds

class SocialOptimum:
    def __init__(self, args, net_data, demand_dict):
        self.args = args
        self.net = net_data
        self.demand = demand_dict
        
        # Hardcoded parameters from original script
        self.C_l_unit = args.pt_cap
        self.S_set = S_SET
        self.W_SPEED = args.walk * MPH_TO_M_PER_MIN
        self.AMOD_WAIT = AMOD_WAIT
        self.base_load_factor = BASE_LOAD_FACTOR

        self.m = gp.Model("Social_Optimum")
        # if args.reg != 2:
        #     self.m.params.NonConvex = 0
        # else:
        #     self.m.params.NonConvex = 1
        self.m.params.NonConvex = 0
        self.m.params.MIPGap = args.mip
        self.m.params.OutputFlag = 1 # Keep output on for the main SO run

        self._build_model()

    def _build_model(self):
        m = self.m
        net = self.net
        demand = self.demand
        args = self.args

        # Variables
        self.x_ap = m.addVars(net['paths_df']['path_id'].unique(), lb=0, name="x_path")
        self.x_a = m.addVars(net['OD_pairs'], lb=1e-6, ub=[demand[od] for od in net['OD_pairs']], name="x_a")
        self.r = m.addVars(net['E_a'], lb=0, name="r_ij")
        self.f = m.addVars(net['E_a'], lb=0, name="f_ij")
        self.v = m.addVars(net['E_a'], lb=[net['b_map'][e] for e in net['E_a']], name="v_total")
        self.v_ratio = m.addVars(net['E_a'], lb=0, name="v_ratio")
        self.v_ratio_pow = m.addVars(net['E_a'], lb=0, name="v_ratio_pow")
        self.x_pt = m.addVars(net['OD_pairs'], lb=1e-6, ub=[demand[od] for od in net['OD_pairs']], name="x_pt")

        self.z = m.addVars(net['L'], self.S_set, vtype=GRB.BINARY, name="z_ls")
        self.s_l = m.addVars(net['L'], lb=min(self.S_set), name="s_l")

        # Transit path choice. An OD with a single candidate keeps today's structure exactly:
        # its path flow *is* x_pt (no y, no t_pt, no big-M rows).
        self.x_pt_p = {}
        self.y = {}
        self.t_pt = {}
        for od in net['OD_pairs']:
            paths = net['pt_paths'][od]
            if len(paths) == 1:
                self.x_pt_p[od, 0] = self.x_pt[od]
                continue
            bounds = [path_time_bounds(p, self.W_SPEED) for p in paths]
            t_lo_min = min(lo for lo, up in bounds)
            self.t_pt[od] = m.addVar(lb=t_lo_min, ub=max(up for lo, up in bounds), name=f"t_pt_{od[0]}_{od[1]}")
            for k in range(len(paths)):
                self.x_pt_p[od, k] = m.addVar(lb=0, ub=demand[od], name=f"x_pt_p_{od[0]}_{od[1]}_{k}")
                self.y[od, k] = m.addVar(vtype=GRB.BINARY, name=f"y_{od[0]}_{od[1]}_{k}")
            m.addConstr(gp.quicksum(self.y[od, k] for k in range(len(paths))) == 1)
            m.addConstr(gp.quicksum(self.x_pt_p[od, k] for k in range(len(paths))) == self.x_pt[od])
            for k, p in enumerate(paths):
                m.addConstr(self.x_pt_p[od, k] <= demand[od] * self.y[od, k])
                t_expr = p['tau_pt'] + p['walking_dist'] / self.W_SPEED + gp.quicksum(
                    (0.5 * 60 / s) * self.z[l, s] for l in p['gamma_odl'] for s in self.S_set)
                m.addConstr(self.t_pt[od] <= t_expr)
                m.addConstr(self.t_pt[od] >= t_expr - (bounds[k][1] - t_lo_min) * (1 - self.y[od, k]))

        # Sparse w indices
        w_indices = []
        for od, paths in net['pt_paths'].items():
            for k, info in enumerate(paths):
                for line in info['gamma_odl']:
                    for s in self.S_set:
                        w_indices.append((od, k, line, s))
        self.w = m.addVars(w_indices, lb=0, ub=[demand[od] for od, k, l, s in w_indices], name="w_sparse")

        # 1. Flow on road and Travel Time
        for e in net['E_a']:
            p_on_e = net['paths_df'][(net['paths_df']['u'] == e[0]) & (net['paths_df']['v'] == e[1])]['path_id'].unique()
            m.addConstr(self.f[e] == self.r[e] + self.x_ap.sum(p_on_e))
            m.addConstr(self.v[e] == net['b_map'][e] + self.f[e])
            m.addConstr(self.v_ratio[e] == self.v[e] / net['Cap_map'][e])
            m.addGenConstrPow(self.v_ratio[e], self.v_ratio_pow[e], args.beta + 1)

        # 2. Flow Conservation
        for node in net['road_nodes']:
            rebal_in = self.r.sum('*', node)
            rebal_out = self.r.sum(node, '*')
            pass_out = gp.quicksum(self.x_a[o, d] for o, d in net['OD_pairs'] if net['zone_to_node'].get(o) == node)
            pass_in = gp.quicksum(self.x_a[o, d] for o, d in net['OD_pairs'] if net['zone_to_node'].get(d) == node)
            m.addConstr(rebal_in + pass_in == rebal_out + pass_out, name=f"flow_bal_{node}")

        # 3. Demand split
        for od in net['OD_pairs']:
            m.addConstr(self.x_a[od] + self.x_pt[od] == demand[od])
            rel_paths = net['paths_df'][(net['paths_df']['o_zone'] == od[0]) & (net['paths_df']['d_zone'] == od[1])]['path_id'].unique()
            m.addConstr(self.x_a[od] == self.x_ap.sum(rel_paths))

        # 4. Transit Capacity & Frequency
        for l in net['L']:
            m.addConstr(self.z.sum(l, '*') == 1)
            m.addConstr(self.s_l[l] == gp.quicksum(s * self.z[l, s] for s in self.S_set))
            for u_s, v_s in net['lines_dict'][l]:
                pass_on_seg = gp.quicksum(self.x_pt_p[odk] for odk in net['seg_to_odp'].get((u_s, v_s, l), []))
                m.addConstr(pass_on_seg <= (1-self.base_load_factor) * self.C_l_unit * self.s_l[l])

        # 5. Linearization
        for (od, k, l, s) in self.w.keys():
            m.addConstr(self.w[od, k, l, s] <= self.x_pt_p[od, k])
            m.addConstr(self.w[od, k, l, s] <= demand[od] * self.z[l, s])
            m.addConstr(self.w[od, k, l, s] >= self.x_pt_p[od, k] - demand[od] * (1 - self.z[l, s]))

        # Objective formulation
        road_time_cost = gp.quicksum(net['T0_map'][e] * (self.v[e] + args.alpha * net['Cap_map'][e] * self.v_ratio_pow[e]) for e in net['E_a']) - net['baseline_road_cost']
        amod_wait = gp.quicksum(self.AMOD_WAIT * self.x_a[od] for od in net['OD_pairs'])
        transit_time_fixed = gp.quicksum(self.x_pt_p[od, k] * p['tau_pt'] for od, paths in net['pt_paths'].items() for k, p in enumerate(paths))
        transit_walk_fixed = gp.quicksum(self.x_pt_p[od, k] * p['walking_dist']/self.W_SPEED for od, paths in net['pt_paths'].items() for k, p in enumerate(paths))
        transit_waiting = 0.5 * 60 * gp.quicksum((1/s) * self.w[od, k, l, s] for (od, k, l, s) in self.w.keys())
        amod_op_cost = gp.quicksum(args.ca * self.f[u, v] * (net['dist_map'][u, v] / 1000.0) for u, v in net['E_a'])
        op_cost = gp.quicksum(args.cl * self.s_l[l] for l in net['L'])
        
        obj = args.vot * road_time_cost + args.vot1 * (transit_time_fixed + transit_waiting + transit_walk_fixed + amod_wait) + amod_op_cost + op_cost

        # Regularization handlers
        if args.reg == 0:
            m.setObjective(obj, GRB.MINIMIZE)
        elif args.reg == 1:
            m.setObjective(obj + (1.0/args.theta) * gp.quicksum(self.x_a[od]**2 + self.x_pt[od]**2 for od in net['OD_pairs']), GRB.MINIMIZE)
        elif args.reg == 2:
            entropy_terms = []
            for od in net['OD_pairs']:
                ent_a = m.addVar(lb=-1.0)
                ent_pt = m.addVar(lb=-1.0)

                # Generate coordinates for the PWL approximation
                max_d = max(1.0, demand[od]) 
                x_pts = np.geomspace(1e-6, max_d, num=30).tolist()
                y_pts = [p * np.log(p) for p in x_pts]
                m.addGenConstrPWL(self.x_a[od], ent_a, x_pts, y_pts)
                m.addGenConstrPWL(self.x_pt[od], ent_pt, x_pts, y_pts)
                entropy_terms.append(ent_a + ent_pt)
            m.setObjective(obj + (1.0/args.theta) * gp.quicksum(entropy_terms), GRB.MINIMIZE)

    def solve(self):
        self.m.optimize()
        
        if self.m.status == GRB.OPTIMAL:
            # Selected transit path per OD (single-candidate ODs are always on path 0)
            self.path_choice = {}
            for od in self.net['OD_pairs']:
                n_p = len(self.net['pt_paths'][od])
                self.path_choice[od] = next((k for k in range(n_p) if self.y[od, k].X > 0.5), 0) if n_p > 1 else 0

            # Package the Transit Operator's target strategy
            target_transit_strategy = {
                'frequencies': {l: self.s_l[l].X for l in self.net['L']},
                'fare': TRANSIT_FARE, # Fixed PT fare
                'path_choice': self.path_choice
            }
            
            # Package the AMoD Operator's target strategy (to use as hot-start)
            target_amod_strategy = {
                'f': {e: self.f[e].X for e in self.net['E_a']},
                'r': {e: self.r[e].X for e in self.net['E_a']}
            }
            
            return self.m.ObjVal, target_transit_strategy, target_amod_strategy
        else:
            print("Social Optimum Infeasible!")
            return None, None, None
    
    def get_results_dataframe(self):
        """
        Recover AMoD price and post-process solved model to get statistics.
        """
        if self.m.status != GRB.OPTIMAL:
            print("Model not solved to optimality. Cannot generate results.")
            return None

        net = self.net
        args = self.args
        
        # Calculate congested times and transit wait times
        congested_time_map = {}
        for e in net['E_a']:
            v_total = self.v[e].X
            ratio = v_total / net['Cap_map'][e]
            congested_time_map[e] = net['T0_map'][e] * (1 + args.alpha * (ratio**args.beta))
            
        line_wait_map = {l: (0.5 * 60 / self.s_l[l].X) if self.s_l[l].X > 0 else 0 for l in net['L']}

        detailed_summary = []
        for od in net['OD_pairs']:
            rel_paths = net['paths_df'][(net['paths_df']['o_zone'] == od[0]) & (net['paths_df']['d_zone'] == od[1])]['path_id'].unique()
            amod_flow = self.x_a[od].X
            
            used_paths_count = sum(1 for p in rel_paths if self.x_ap[p].X > 1e-6)
            best_p = sorted(rel_paths)[0] if len(rel_paths) > 0 else None
            
            if amod_flow > 1e-6:
                pot_amod_dist = sum(self.x_ap[p].X * sum(net['dist_map'][r.u, r.v] for r in net['paths_df'][net['paths_df']['path_id']==p].itertuples()) 
                                for p in rel_paths) / (amod_flow * 1000.0)
                pot_amod_time = sum(self.x_ap[p].X * (sum(congested_time_map[r.u, r.v] for r in net['paths_df'][net['paths_df']['path_id']==p].itertuples()) + self.AMOD_WAIT) 
                                for p in rel_paths) / amod_flow
            elif best_p:
                p_edges = net['paths_df'][net['paths_df']['path_id'] == best_p]
                pot_amod_dist = sum(net['dist_map'][r.u, r.v] for r in p_edges.itertuples()) / 1000.0
                pot_amod_time = sum(congested_time_map[r.u, r.v] for r in p_edges.itertuples()) + self.AMOD_WAIT
            else:
                pot_amod_dist, pot_amod_time = 0, 0

            k_sel = self.path_choice[od]
            path = net['pt_paths'][od][k_sel]
            pot_transit_wait = sum(line_wait_map[l] for l in path['gamma_odl'])
            walking_time = path['walking_dist'] / self.W_SPEED
            pot_transit_total_time = path['tau_pt'] + walking_time + pot_transit_wait

            detailed_summary.append({
                'O': od[0], 'D': od[1],
                'Demand': self.demand[od],
                'AMoD_Flow': round(amod_flow, 2),
                'Transit_Flow': round(self.x_pt[od].X, 2),
                'AMoD_Paths_Used': used_paths_count,
                'Pot_AMoD_Time_min': round(pot_amod_time, 2),
                'Pot_AMoD_Dist_km': round(pot_amod_dist, 2),
                'Pot_Transit_Wait_min': round(pot_transit_wait, 2),
                'Pot_Transit_Total_min': round(pot_transit_total_time, 2),
                'Transit_Path': k_sel,
                'Transit_Paths_Avail': len(net['pt_paths'][od])
            })

        df_results = pd.DataFrame(detailed_summary)
        
        # Recover the prices using your pricing.py function
        # Note: Make sure to pass the transit fare properly (using your constant or parameter)
        fare = TRANSIT_FARE
        df_results = recover_amod_price(df_results, args.vot1, fare)
        
        return df_results

    def get_summary(self):
        net = self.net
        args = self.args
        
        # 1. Transit frequency and operation cost
        transit_summary = []
        for l in net['L']:
            line_passengers = sum(self.x_pt_p[od, k].X for od, paths in net['pt_paths'].items()
                                  for k, p in enumerate(paths) if l in p['gamma_odl'])
            transit_summary.append({
                'Line_ID': l,
                'Frequency_vph': round(self.s_l[l].X, 2),
                'Total_Passengers': round(line_passengers, 2),
            })
        
        df_transit = pd.DataFrame(transit_summary)
        total_transit_op_cost = sum(args.cl * self.s_l[l].X for l in net['L'])
        avg_freq = df_transit['Frequency_vph'].mean()
        
        # 2. AMoD distances
        total_rebal = sum(self.r[e].X * net['dist_map'][e] for e in net['E_a']) / 1000
        total_pass_dist = sum(self.x_ap[p].X * sum(net['dist_map'][r.u, r.v] for r in net['paths_df'][net['paths_df']['path_id']==p].itertuples()) 
                             for p in self.x_ap.keys() if self.x_ap[p].X > 0) / 1000
        
        # 3. Mode split
        df_results = self.get_results_dataframe()
        total_amod = df_results['AMoD_Flow'].sum()
        total_transit = df_results['Transit_Flow'].sum()
        total_demand = df_results['Demand'].sum()
        
        amod_share = (total_amod / total_demand) * 100 if total_demand > 0 else 0
        transit_share = (total_transit / total_demand) * 100 if total_demand > 0 else 0
        
        # 4. Background traffic impact
        total_bg_increase_min = 0
        for e in net['E_a']:
            v_final = self.v[e].X
            t_congested = net['T0_map'][e] * (1 + args.alpha * ((v_final / net['Cap_map'][e]) ** args.beta))
            t_baseline = net['T0_map'][e] * (1 + args.alpha * ((net['b_map'][e] / net['Cap_map'][e]) ** args.beta))
            edge_increase = net['b_map'][e] * (t_congested - t_baseline)
            total_bg_increase_min += edge_increase
        
        # 5. Calculate objective components for social cost
        congested_time_map = {}
        for e in net['E_a']:
            v_total = self.v[e].X
            ratio = v_total / net['Cap_map'][e]
            congested_time_map[e] = net['T0_map'][e] * (1 + args.alpha * (ratio**args.beta))
        
        road_time_cost = sum(net['T0_map'][e] * (self.v[e].X + args.alpha * net['Cap_map'][e] * (self.v_ratio_pow[e].X)) for e in net['E_a']) - net['baseline_road_cost']
        amod_wait = self.AMOD_WAIT * total_amod
        transit_time_fixed = sum(self.x_pt_p[od, k].X * p['tau_pt'] for od, paths in net['pt_paths'].items() for k, p in enumerate(paths))
        transit_walk_fixed = sum(self.x_pt_p[od, k].X * p['walking_dist']/self.W_SPEED for od, paths in net['pt_paths'].items() for k, p in enumerate(paths))
        transit_waiting = 0.5 * 60 * sum((1/s) * self.w[od, k, l, s].X for (od, k, l, s) in self.w.keys())
        amod_op_cost = sum(args.ca * self.f[u, v].X * (net['dist_map'][u, v] / 1000.0) for u, v in net['E_a'])
        op_cost = sum(args.cl * self.s_l[l].X for l in net['L'])
        
        total_physical_cost = (
            args.vot * road_time_cost + 
            args.vot1 * (transit_time_fixed + transit_waiting + transit_walk_fixed + amod_wait) + 
            amod_op_cost + 
            op_cost
        )
        
        # 6. Transit wait time average
        avg_wait = df_results['Pot_Transit_Wait_min'].mean()
        
        # Create summary dataframe
        run_summary = {
            "BL": args.bl,
            "K_l": args.cl,
            "C_ij": args.ca,
            "Num_Path": args.num_path,
            # Social Cost (main metric)
            "Total_Social_Cost": round(total_physical_cost, 2),
            
            # Mode Split
            "Total_Demand": total_demand,
            "AMoD_Flow": round(total_amod, 2),
            "Transit_Flow": round(total_transit, 2),
            "AMoD_Share_pct": round(amod_share, 2),
            "Transit_Share_pct": round(transit_share, 2),

            # Supply Side Metrics
            "Avg_Transit_Freq": round(avg_freq, 2),
            "Avg_Wait_Time_min": round(avg_wait, 2),
            "Transit_Op_Cost": round(total_transit_op_cost, 2),
            "AMoD_Op_Cost": round(amod_op_cost, 2),
            
            # Road/Congestion Metrics
            "Rebalancing_Dist_km": round(total_rebal, 2),
            "AMoD_Pax_Dist_km": round(total_pass_dist, 2),
            "Rebal_Ratio": round(total_rebal / total_pass_dist, 3) if total_pass_dist > 0 else 0,
            "BG_Delay_Total_min": round(total_bg_increase_min, 2),
        }

        return pd.DataFrame([run_summary])


def evaluate_baseline_social_cost(args, net_data, demand, amod_state, freq_dict, path_choice=None):
    """Social-cost summary at a fixed AMoD-best-response solution with externally-
    provided transit frequencies (the no-subsidy baseline). Returns the same
    DataFrame schema as `SocialOptimum.get_summary()`, computed with the exact
    same formulas — keep this function in sync with `get_summary` line-for-line.

    amod_state: dict with keys 'f', 'r', 'x_a', 'x_ap' from
                AMoD_BestResponse.solve_sca() (plain dicts of solved values).
    freq_dict:  {line_id: frequency_vph} covering every line in net_data['L'].
    path_choice: optional {od: k} to freeze the transit path (e.g. the SO's `y`);
                otherwise the min-time candidate under `freq_dict` is used.
    """
    net = net_data
    f, r, x_a, x_ap = amod_state['f'], amod_state['r'], amod_state['x_a'], amod_state['x_ap']

    # Constants — match SocialOptimum.__init__ defaults
    W_SPEED = args.walk * MPH_TO_M_PER_MIN

    # Transit path per OD, mirroring the SO's `y` (see SocialOptimum._build_model).
    prefer = path_choice or {}
    chosen = {}
    for od in net['OD_pairs']:
        k, _ = choose_path(net['pt_paths'][od], freq_dict, W_SPEED, prefer=prefer.get(od))
        chosen[od] = net['pt_paths'][od][k]

    # AMoD-BR enforces x_a + x_pt == demand (amod_br.py:128); derive x_pt accordingly.
    x_pt = {od: demand[od] - x_a[od] for od in net['OD_pairs']}

    # Mirror `v[e] == b_map[e] + f[e]` (social_optimum.py:90) and GenConstrPow on v/Cap (line 92).
    v = {e: net['b_map'][e] + f[e] for e in net['E_a']}
    v_ratio_pow = {e: (v[e] / net['Cap_map'][e]) ** (args.beta + 1) for e in net['E_a']}

    # 1. Transit frequency and operation cost (mirrors get_summary lines 254-267)
    transit_summary = []
    for l in net['L']:
        line_passengers = sum(x_pt[od] for od in net['OD_pairs'] if l in chosen[od]['gamma_odl'])
        transit_summary.append({
            'Line_ID': l,
            'Frequency_vph': round(freq_dict[l], 2),
            'Total_Passengers': round(line_passengers, 2),
        })
    df_transit = pd.DataFrame(transit_summary)
    total_transit_op_cost = sum(args.cl * freq_dict[l] for l in net['L'])
    avg_freq = df_transit['Frequency_vph'].mean()

    # 2. AMoD distances (mirrors lines 269-272)
    total_rebal = sum(r[e] * net['dist_map'][e] for e in net['E_a']) / 1000
    total_pass_dist = sum(
        x_ap[p] * sum(net['dist_map'][rec.u, rec.v]
                      for rec in net['paths_df'][net['paths_df']['path_id'] == p].itertuples())
        for p in x_ap.keys() if x_ap[p] > 0
    ) / 1000

    # 3. Mode split (mirrors lines 274-281, computed directly to avoid building results df)
    total_amod = sum(x_a[od] for od in net['OD_pairs'])
    total_transit = sum(x_pt[od] for od in net['OD_pairs'])
    total_demand = sum(demand[od] for od in net['OD_pairs'])
    amod_share = (total_amod / total_demand) * 100 if total_demand > 0 else 0
    transit_share = (total_transit / total_demand) * 100 if total_demand > 0 else 0

    # 4. Background traffic impact (mirrors lines 283-290)
    total_bg_increase_min = 0
    for e in net['E_a']:
        v_final = v[e]
        t_congested = net['T0_map'][e] * (1 + args.alpha * ((v_final / net['Cap_map'][e]) ** args.beta))
        t_baseline = net['T0_map'][e] * (1 + args.alpha * ((net['b_map'][e] / net['Cap_map'][e]) ** args.beta))
        edge_increase = net['b_map'][e] * (t_congested - t_baseline)
        total_bg_increase_min += edge_increase

    # 5. Social-cost components (mirrors lines 292-312)
    # NOTE: in SO the transit-wait term is `0.5*60 * sum_{(od,k,l,s)} (1/s) * w[od,k,l,s]`.
    # The SO constraints force `w[od,k,l,s] = x_pt_p[od,k] * z[l,s]` with exactly one s active
    # per line, so summed over s it collapses to `x_pt_p[od,k] * (1/s_l)`, and x_pt_p is zero
    # off the chosen path. The form below is therefore algebraically identical to the SO
    # expression when freq_dict = {l: s_l.X} and path_choice = the SO's `y`.
    road_time_cost = sum(net['T0_map'][e] * (v[e] + args.alpha * net['Cap_map'][e] * v_ratio_pow[e]) for e in net['E_a']) - net['baseline_road_cost']
    amod_wait = AMOD_WAIT * total_amod
    transit_time_fixed = sum(x_pt[od] * chosen[od]['tau_pt'] for od in net['OD_pairs'])
    transit_walk_fixed = sum(x_pt[od] * chosen[od]['walking_dist'] / W_SPEED for od in net['OD_pairs'])
    transit_waiting = 0.5 * 60 * sum(
        x_pt[od] * sum(1.0 / freq_dict[l] for l in chosen[od]['gamma_odl'] if freq_dict[l] > 0)
        for od in net['OD_pairs']
    )
    amod_op_cost = sum(args.ca * f[u, v_edge] * (net['dist_map'][u, v_edge] / 1000.0) for u, v_edge in net['E_a'])
    op_cost = sum(args.cl * freq_dict[l] for l in net['L'])

    total_physical_cost = (
        args.vot * road_time_cost +
        args.vot1 * (transit_time_fixed + transit_waiting + transit_walk_fixed + amod_wait) +
        amod_op_cost +
        op_cost
    )

    # 6. Per-OD potential transit wait (mirrors get_results_dataframe + the summary's avg_wait)
    line_wait_map = {l: (0.5 * 60 / freq_dict[l]) if freq_dict[l] > 0 else 0 for l in net['L']}
    pot_transit_wait_min = [sum(line_wait_map[l] for l in chosen[od]['gamma_odl']) for od in net['OD_pairs']]
    avg_wait = float(np.mean(pot_transit_wait_min)) if pot_transit_wait_min else 0.0

    run_summary = {
        "BL": args.bl,
        "K_l": args.cl,
        "C_ij": args.ca,
        "Num_Path": args.num_path,
        "Total_Social_Cost": round(total_physical_cost, 2),
        "Total_Demand": total_demand,
        "AMoD_Flow": round(total_amod, 2),
        "Transit_Flow": round(total_transit, 2),
        "AMoD_Share_pct": round(amod_share, 2),
        "Transit_Share_pct": round(transit_share, 2),
        "Avg_Transit_Freq": round(avg_freq, 2),
        "Avg_Wait_Time_min": round(avg_wait, 2),
        "Transit_Op_Cost": round(total_transit_op_cost, 2),
        "AMoD_Op_Cost": round(amod_op_cost, 2),
        "Rebalancing_Dist_km": round(total_rebal, 2),
        "AMoD_Pax_Dist_km": round(total_pass_dist, 2),
        "Rebal_Ratio": round(total_rebal / total_pass_dist, 3) if total_pass_dist > 0 else 0,
        "BG_Delay_Total_min": round(total_bg_increase_min, 2),
    }
    return pd.DataFrame([run_summary])