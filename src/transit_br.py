import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

from src.constants import S_SET, TRANSIT_FARE, MPH_TO_M_PER_MIN, AMOD_WAIT, BASE_LOAD_FACTOR
from src.transit_paths import choose_path, path_time_bounds

class TransitBestResponse:
    def __init__(self, args, net_data, demand_dict, amod_so_solution, transit_so_solution):

        self.args = args
        self.net = net_data
        self.demand = demand_dict
        self.transit_so = transit_so_solution

        self.W_SPEED = args.walk * MPH_TO_M_PER_MIN
        self.C_l_unit = args.pt_cap

        # Calculate fixed AMoD utility V_a based on Social Optimum path assignments
        self.v_a = self._calculate_amod_utility(amod_so_solution)

        self.m = gp.Model("Transit_BestResponse")
        self.m.params.MIPGap = args.mip
        self.m.params.NonConvex = 0

        self.m.params.MIPFocus = 3     # Focus heavily on proving the bound
        self.m.params.Cuts = 2         # Aggressive cut generation for Big-M
        self.m.params.Method = 2       # Use the Barrier method for the root LP
        self.m.params.Threads = 0      # Use all available CPU cores

        self._build_model()

    def _calculate_amod_utility(self, amod_solution):
        V_a = {}
        # Calculate congested edge times based on SO flows
        t_e = {}
        for e in self.net['E_a']:
            v_total = self.net['b_map'][e] + amod_solution['f'][e]
            ratio = v_total / self.net['Cap_map'][e]
            t_e[e] = self.net['T0_map'][e] * (1 + self.args.alpha * (ratio ** self.args.beta))

        for od in self.net['OD_pairs']:
            rel_paths = self.net['paths_df'][(self.net['paths_df']['o_zone'] == od[0]) &
                                             (self.net['paths_df']['d_zone'] == od[1])]['path_id'].unique()
            x_a_total = amod_solution['x_a'][od]

            # Calculate weighted expected travel time acorss all used paths
            if x_a_total > 1e-6:
                expected_time = 0
                for p in rel_paths:
                    x_ap = amod_solution['x_ap'].get(p,0)
                    if x_ap > 1e-6:
                        p_edges = self.net['paths_df'][self.net['paths_df']['path_id'] == p]
                        t_p = sum(t_e[(r.u, r.v)] for r in p_edges.itertuples()) + AMOD_WAIT
                        # Probability of path given AMoD choice
                        expected_time += (x_ap / x_a_total) * t_p
            else:
                # Fallback utility if AMoD flow was 0 in SO (Evaluate shortest path)
                best_p = sorted(rel_paths)[0] if len(rel_paths) > 0 else None
                if best_p:
                    p_edges = self.net['paths_df'][self.net['paths_df']['path_id'] == best_p]
                    expected_time = sum(t_e[(r.u, r.v)] for r in p_edges.itertuples()) + AMOD_WAIT
                else:
                    expected_time = 0
            pi_a = amod_solution['pi_a'][od]
            # Calculate final AMoD generalized utility
            V_a[od] = -self.args.vot * expected_time - pi_a

        return V_a
    
    def _build_model(self):
        m = self.m
        net = self.net
        demand = self.demand

        # Variables
        self.x_pt = m.addVars(net['OD_pairs'], lb=1e-6, ub=[demand[od] for od in net['OD_pairs']], name="x_pt")
        self.z = m.addVars(net['L'], S_SET, vtype=GRB.BINARY, name="z_ls")
        self.s_l = m.addVars(net['L'], lb=min(S_SET), name="s_l")

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
                    (30.0 / s) * self.z[l, s] for l in p['gamma_odl'] for s in S_SET)
                m.addConstr(self.t_pt[od] <= t_expr)
                m.addConstr(self.t_pt[od] >= t_expr - (bounds[k][1] - t_lo_min) * (1 - self.y[od, k]))

        # ---------------------------
        # Warm start with so solution
        # ---------------------------
        for l in net['L']:
            so_freq = self.transit_so['frequencies'][l]
            self.s_l[l].Start = so_freq
            for s in S_SET:
                # If this frequency was chosen in SO, set binary start to 1.0
                self.z[l, s].Start = 1.0 if s == so_freq else 0.0

        so_path_choice = self.transit_so.get('path_choice', {})
        for od in net['OD_pairs']:
            self.x_pt[od].Start = self.transit_so['x_pt'][od]
            if len(net['pt_paths'][od]) > 1:
                k_so = so_path_choice.get(od, 0)
                for k in range(len(net['pt_paths'][od])):
                    self.y[od, k].Start = 1.0 if k == k_so else 0.0
                    self.x_pt_p[od, k].Start = self.transit_so['x_pt'][od] if k == k_so else 0.0

        # Auxiliary variables for Big-M linearization (u = x_pt_p * z)
        u_indices = []
        for od, paths in net['pt_paths'].items():
            for k, info in enumerate(paths):
                for line in info['gamma_odl']:
                    for s in S_SET:
                        u_indices.append((od, k, line, s))
        self.u = m.addVars(u_indices, lb=0, ub=[demand[od] for od, k, l, s in u_indices], name="u_linearized")

        # 1. Frequency Selection
        for l in net['L']:
            m.addConstr(self.z.sum(l, '*') == 1)
            m.addConstr(self.s_l[l] == gp.quicksum(s * self.z[l, s] for s in S_SET))

        # 2. Big-M Exact Linearization for u
        for (od, k, l, s) in self.u.keys():
            m.addConstr(self.u[od, k, l, s] <= self.x_pt_p[od, k])
            m.addConstr(self.u[od, k, l, s] <= demand[od] * self.z[l, s])
            m.addConstr(self.u[od, k, l, s] >= self.x_pt_p[od, k] - demand[od] * (1 - self.z[l, s]))

        # 3. Convex Entropy Term H(x) = x ln(x) - x ln(d-x)
        self.H_x = m.addVars(net['OD_pairs'], lb=-GRB.INFINITY, name="Entropy_H")
        for od in net['OD_pairs']:
            d = demand[od]
            if d > 1e-2:
                self.x_pt[od].ub = d - 1e-6
                half_pts = np.geomspace(1e-6, d / 2.0, num=20)
                right_pts = d - half_pts[::-1] 
                x_pts = np.unique(np.concatenate((half_pts, right_pts))).tolist()           
                # Tangent lines for purely continuous outer approximation
                for x_val in x_pts:
                    # Nudge slightly away from exact 0 and d to prevent division/log by zero
                    x_safe = max(1e-5, min(x_val, d - 1e-5))
                    
                    y_val = x_safe * np.log(x_safe) - x_safe * np.log(d - x_safe)
                    
                    # Exact derivative for the tangent slope
                    grad = np.log(x_safe) + 1 - np.log(d - x_safe) + (x_safe / (d - x_safe))
                    intercept = y_val - grad * x_safe
                    
                    m.addConstr(self.H_x[od] >= grad * self.x_pt[od] + intercept)
            else:
                m.addConstr(self.x_pt[od] == 1e-6)
                m.addConstr(self.H_x[od] == 0)

        # 4. Relaxed Logit Constraint
        for od in net['OD_pairs']:
            paths = net['pt_paths'][od]

            # Fixed transit time (walk + in-vehicle) is now path-dependent; the fare is not
            fixed_term = gp.quicksum(self.x_pt_p[od, k] * (p['tau_pt'] + p['walking_dist'] / self.W_SPEED)
                                     for k, p in enumerate(paths))

            # Linear combination of wait time using u variables
            wait_term = gp.quicksum((30.0 / s) * self.u[od, k, l, s]
                                    for k, p in enumerate(paths) for l in p['gamma_odl'] for s in S_SET)

            m.addConstr(self.H_x[od] + self.args.vot1 * (fixed_term + wait_term)
                        + self.x_pt[od] * (TRANSIT_FARE + self.v_a[od]) <= 0)

        # 5. Transit Capacity Constraint (The physical link between ridership and frequency)
        for l in net['L']:
            for u_s, v_s in net['lines_dict'][l]:
                # Sum all passengers whose chosen path uses this specific line segment
                pass_on_seg = gp.quicksum(self.x_pt_p[odk] for odk in net['seg_to_odp'].get((u_s, v_s, l), []))

                # The total flow on the segment cannot exceed the hourly capacity of the trains
                m.addConstr(pass_on_seg <= (1 - BASE_LOAD_FACTOR) * self.C_l_unit * self.s_l[l])

        # 6. Objective: Maximize (Fare + Social Bonus) * ridership - Op Cost
        revenue_and_bonus = gp.quicksum((TRANSIT_FARE + self.args.omega) * self.x_pt[od] for od in net['OD_pairs'])
        op_cost = gp.quicksum(self.args.cl * self.s_l[l] for l in net['L'])
        m.setObjective(revenue_and_bonus - op_cost, GRB.MAXIMIZE)

    def solve(self):
        self.m.optimize()
        if self.m.status == GRB.OPTIMAL:
            # Dual upper bound from the relaxation
            profit_ub = self.m.ObjVal
            freq_opt = {l: self.s_l[l].X for l in self.net['L']}
            solver_x_pt = {od:self.x_pt[od].X for od in self.net['OD_pairs']}

            # -----------------------------------
            # Check if relaxation is tight
            # -----------------------------------
            tight_relaxation = True

            for od in self.net['OD_pairs']:
                d = self.demand[od]
                if d <= 1e-6:
                    continue

                paths = self.net['pt_paths'][od]
                # Calculate the exact fixed / wait terms from solver's auxiliary variables
                fixed_term_val = sum(self.x_pt_p[od, k].X * (p['tau_pt'] + p['walking_dist'] / self.W_SPEED)
                                     for k, p in enumerate(paths))
                wait_term_val = sum((30.0 / s) * self.u[od, k, l, s].X
                                    for k, p in enumerate(paths) for l in p['gamma_odl'] for s in S_SET)

                # Evaluate the slack of the inequality constraint
                # --- NEW LOGIC: Calculate the TRUE entropy value ---
                x_val = solver_x_pt[od]
                x_safe = max(1e-5, min(x_val, d - 1e-5)) # Protect against log(0)
                true_H = x_safe * np.log(x_safe) - x_safe * np.log(d - x_safe)

                # Evaluate the true slack of the original non-linear equality
                solver_lhs = (true_H + self.args.vot1 * (fixed_term_val + wait_term_val)
                              + x_val * (TRANSIT_FARE + self.v_a[od]))

                if solver_lhs < -1e-2: # Numerical tolerance for tightness
                    tight_relaxation = False
                    break
            # ---------------------------------------------------------
            # Path 1: relaxation is tight
            # ---------------------------------------------------------
            if tight_relaxation:
                print("\n--- Transit Best Response Results ---")
                print("Relaxation State:            TIGHT")
                print(f"Optimal Profit:              {profit_ub:.2f}")
                print("Suboptimality Gap:           0.00% (Solution is strictly optimal)\n")

                # Selected transit path per OD (single-candidate ODs are always on path 0)
                path_choice = {}
                for od in self.net['OD_pairs']:
                    n_p = len(self.net['pt_paths'][od])
                    path_choice[od] = next((k for k in range(n_p) if self.y[od, k].X > 0.5), 0) if n_p > 1 else 0

                br_strategy = {
                    'frequencies': freq_opt,
                    'fare': TRANSIT_FARE,
                    'x_pt': solver_x_pt,
                    'path_choice': path_choice
                }

                # Return the Upper Bound as the absolute optimal profit
                return profit_ub, br_strategy, True, 0
            # ---------------------------------------------------------
            # Path 2: relaxation is loose (repair frequencie)
            # ---------------------------------------------------------
            else:
                print("\n--- Transit Best Response Results ---")
                print("Relaxation State:            LOOSE (Capacity Binding)")
                print("Repairing frequencies to evaluate true optimality gap...")

                # Calculate Exact Strict Logit Flows for the Repaired Upper Bound
                feasible_freq = freq_opt.copy()
                final_x_pt = {}
                path_choice = {}
                sorted_S = sorted(list(S_SET))
                recovery_iterations = 0

                while True:
                    recovery_iterations += 1

                    # Exact non-linear logit demand under current frequencies. Frequencies change
                    # between iterations, so the min-time candidate is re-selected every time.
                    for od in self.net['OD_pairs']:
                        d = self.demand[od]
                        k, t_pt = choose_path(self.net['pt_paths'][od], feasible_freq, self.W_SPEED)
                        path_choice[od] = k
                        if d <= 1e-6 or t_pt == float('inf'):
                            final_x_pt[od] = 0.0
                            continue

                        v_pt = -self.args.vot1 * t_pt - TRANSIT_FARE
                        v_a = self.v_a[od]

                        exponent = v_a - v_pt
                        if exponent > 500: final_x_pt[od] = 0
                        elif exponent < -500: final_x_pt[od] = d
                        else: final_x_pt[od] = d / (1 + np.exp(exponent))

                    # Check for capacity violations
                    violated_lines = set()
                    for l in self.net['L']:
                        if feasible_freq[l] < 1e-6: continue

                        for u_s, v_s in self.net['lines_dict'][l]:
                            ods_on_seg = [od for (od, k) in self.net['seg_to_odp'].get((u_s, v_s, l), []) if path_choice.get(od) == k]
                            pass_on_seg = sum(final_x_pt[od] for od in ods_on_seg)
                            seg_cap = (1- BASE_LOAD_FACTOR) * self.C_l_unit * feasible_freq[l]

                            if pass_on_seg > seg_cap + 1e-4:
                                violated_lines.add(l)
                    if not violated_lines:
                        print(f"Feasibility Recovered in {recovery_iterations} iterations.")
                        break

                    # Increase frequencies for violated lines
                    upgraded_any = False
                    for l in violated_lines:
                        current_s = feasible_freq[l]
                        larger_s = [s for s in sorted_S if s > current_s]
                        if larger_s:
                            feasible_freq[l] = larger_s[0]
                            upgraded_any = True
                    
                    if not upgraded_any:
                        print("Warning: Max frequency reached. Applying strict proportional truncation to maxed-out lines.")
                        # Store the tightest bottleneck scale factor for each OD pair
                        od_scale_factors = {od: 1.0 for od in self.net['OD_pairs']}
                        
                        for l in violated_lines:
                            for u_s, v_s in self.net['lines_dict'][l]:
                                ods_on_seg = [od for (od, k) in self.net['seg_to_odp'].get((u_s, v_s, l), []) if path_choice.get(od) == k]
                                pass_on_seg = sum(final_x_pt[od] for od in ods_on_seg)
                                seg_cap = (1 - BASE_LOAD_FACTOR) * self.C_l_unit * feasible_freq[l]
                                
                                if pass_on_seg > seg_cap + 1e-4:
                                    scale_factor = seg_cap / pass_on_seg
                                    # Update the OD pair with the most restrictive bottleneck it encounters
                                    for od in ods_on_seg:
                                        od_scale_factors[od] = min(od_scale_factors[od], scale_factor)
                        
                        # Apply the worst-case scaling uniformly
                        for od in self.net['OD_pairs']:
                            if od_scale_factors[od] < 1.0:
                                final_x_pt[od] *= od_scale_factors[od]
                        break
                
                # Evaluate feasible primal lower bound
                br_strategy = {
                    'frequencies': feasible_freq,
                    'fare': TRANSIT_FARE,
                    'x_pt': final_x_pt,
                    'path_choice': path_choice
                }
                profit_lb = self.evaluate(br_strategy)

                # Print Final Gap Summary
                print(f"Upper Bound (relaxed):  {profit_ub:.2f}")
                print(f"Primal Lower Bound (True):   {profit_lb:.2f}")
                
                gap = ((profit_ub - profit_lb) / abs(profit_ub)) * 100 if profit_ub != 0 else 0
                print(f"Suboptimality Gap:           {gap:.2f}%\n")
                
                # Returns the tight_relaxation flag as requested!
                return profit_lb, br_strategy, False, gap

        else:
            print("Transit Best Response Failed to Solve.")
            return None, None, False, None
        
    def get_results_dataframe(self, transit_res, amod_so_solution):
            """
            Return OD-level statistics for the Transit Best Response.
            transit_res: dict containing 'frequencies' and 'x_pt' from the Transit BR output.
            amod_so_solution: dict containing social optimum AMoD solution ('f', 'x_a', 'x_ap', 'pi_a').
            """
            
            # Calculate congested edge times based entirely on FIXED AMoD flows
            t_e = {}
            for e in self.net['E_a']:
                v_total = self.net['b_map'][e] + amod_so_solution['f'][e]
                ratio = v_total / self.net['Cap_map'][e]
                t_e[e] = self.net['T0_map'][e] * (1 + self.args.alpha * (ratio ** self.args.beta))

            line_wait_map = {l: (0.5 * 60 / transit_res['frequencies'][l]) if transit_res['frequencies'][l] > 0 else 0 for l in self.net['L']}

            detailed_summary = []
            for od in self.net['OD_pairs']:
                d = self.demand[od]
                transit_flow = transit_res['x_pt'][od]
                
                # Recalculate AMoD flow dynamically
                amod_flow_new = d - transit_flow
                
                rel_paths = self.net['paths_df'][(self.net['paths_df']['o_zone'] == od[0]) & (self.net['paths_df']['d_zone'] == od[1])]['path_id'].unique()
                
                # Use the FIXED AMoD solution strictly for calculating routing proportions 
                fixed_x_a = amod_so_solution['x_a'][od]
                best_p = sorted(rel_paths)[0] if len(rel_paths) > 0 else None

                if fixed_x_a > 1e-6:
                    pot_amod_dist = sum(amod_so_solution['x_ap'].get(p, 0) * sum(self.net['dist_map'][r.u, r.v] for r in self.net['paths_df'][self.net['paths_df']['path_id']==p].itertuples()) for p in rel_paths) / (fixed_x_a * 1000.0)
                    pot_amod_time = sum(amod_so_solution['x_ap'].get(p, 0) * (sum(t_e[r.u, r.v] for r in self.net['paths_df'][self.net['paths_df']['path_id']==p].itertuples()) + AMOD_WAIT) for p in rel_paths) / fixed_x_a
                elif best_p:
                    p_edges = self.net['paths_df'][self.net['paths_df']['path_id'] == best_p]
                    pot_amod_dist = sum(self.net['dist_map'][r.u, r.v] for r in p_edges.itertuples()) / 1000.0
                    pot_amod_time = sum(t_e[r.u, r.v] for r in p_edges.itertuples()) + AMOD_WAIT
                else:
                    pot_amod_dist, pot_amod_time = 0, 0

                k_sel = transit_res.get('path_choice', {}).get(od)
                if k_sel is None:
                    k_sel, _ = choose_path(self.net['pt_paths'][od], transit_res['frequencies'], self.W_SPEED)
                path = self.net['pt_paths'][od][k_sel]
                pot_transit_wait = sum(line_wait_map[l] for l in path['gamma_odl'])
                walking_time = path['walking_dist'] / self.W_SPEED
                pot_transit_total_time = path['tau_pt'] + walking_time + pot_transit_wait

                detailed_summary.append({
                    'O': od[0], 'D': od[1],
                    'Demand': d,
                    'AMoD_Flow': round(amod_flow_new, 2),
                    'Transit_Flow': round(transit_flow, 2),
                    'Pot_AMoD_Time_min': round(pot_amod_time, 2),
                    'Pot_AMoD_Dist_km': round(pot_amod_dist, 2),
                    # Price strictly remains fixed under transit deviation
                    'AMoD_Price': round(amod_so_solution['pi_a'].get(od, 0), 2),
                    'Pot_Transit_Wait_min': round(pot_transit_wait, 2),
                    'Pot_Transit_Total_min': round(pot_transit_total_time, 2),
                    'Transit_Path': k_sel
                })

            return pd.DataFrame(detailed_summary)
            
    def evaluate(self, transit_solution):
        """
        Evalue transit utility (objective) at a given solution. Requred input contains:
        - 'x_pt': dict of Transit passenger flow per OD pair
        - 'frequencies': dict of frequency per transit line
        - 'fare': scalar representing the fixed fare
        """

        fare = transit_solution.get('fare', TRANSIT_FARE)
        
        revenue_and_bonus = sum((fare + self.args.omega) * transit_solution['x_pt'][od] for od in self.net['OD_pairs'])
        op_cost = sum(self.args.cl * transit_solution['frequencies'][l] for l in self.net['L'])
        
        objective_value = revenue_and_bonus - op_cost
        return objective_value