import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.constants import S_SET, TRANSIT_FARE, MPH_TO_M_PER_MIN, AMOD_WAIT, BASE_LOAD_FACTOR

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

        # ---------------------------
        # Warm start with so solution
        # ---------------------------
        for l in net['L']:
            so_freq = self.transit_so['frequencies'][l]
            self.s_l[l].Start = so_freq
            for s in S_SET:
                # If this frequency was chosen in SO, set binary start to 1.0
                self.z[l, s].Start = 1.0 if s == so_freq else 0.0

        for od in net['OD_pairs']:
            self.x_pt[od].Start = self.transit_so['x_pt'][od]

        # Auxiliary variables for Big-M linearization (u = x_pt * z)
        u_indices = []
        for od, info in net['opt_data'].items():
            for line in info['gamma_odl']:
                for s in S_SET:
                    u_indices.append((od, line, s))
        self.u = m.addVars(u_indices, lb=0, ub=[demand[od] for od, l, s in u_indices], name="u_linearized")

        # 1. Frequency Selection
        for l in net['L']:
            m.addConstr(self.z.sum(l, '*') == 1)
            m.addConstr(self.s_l[l] == gp.quicksum(s * self.z[l, s] for s in S_SET))

        # 2. Big-M Exact Linearization for u
        for (od, l, s) in self.u.keys():
            m.addConstr(self.u[od, l, s] <= self.x_pt[od])
            m.addConstr(self.u[od, l, s] <= demand[od] * self.z[l, s])
            m.addConstr(self.u[od, l, s] >= self.x_pt[od] - demand[od] * (1 - self.z[l, s]))

        # 3. Convex Entropy Term H(x) = x ln(x) - x ln(d-x)
        self.H_x = m.addVars(net['OD_pairs'], lb=-GRB.INFINITY, name="Entropy_H")
        for od in net['OD_pairs']:
            d = demand[od]
            if d > 1e-2:
                self.x_pt[od].ub = d - 1e-6
                half_pts = np.geomspace(1e-6, d / 2.0, num=10)
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
            opt_data = net['opt_data'][od]
            
            # C_od = Fixed Transit Constants (Fare + Walk Time + In-Vehicle Time)
            walk_time = opt_data['walking_dist'] / self.W_SPEED
            C_od = self.args.vot1 * (opt_data['tau_pt'] + walk_time) + TRANSIT_FARE
            
            # Linear combination of wait time using u variables
            wait_term = gp.quicksum((30.0 / s) * self.u[od, l, s] 
                                    for l in opt_data['gamma_odl'] for s in S_SET)
            
            m.addConstr(self.H_x[od] + self.args.vot1 * wait_term + self.x_pt[od] * (C_od + self.v_a[od]) <= 0)

        # 5. Transit Capacity Constraint (The physical link between ridership and frequency)
        for l in net['L']:
            for u_s, v_s in net['lines_dict'][l]:
                # Sum all passengers whose path uses this specific line segment
                pass_on_seg = gp.quicksum(self.x_pt[od] for od in net['OD_pairs'] if (u_s, v_s, l) in net['opt_data'][od]['eta_e_od'])
                
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

            # Feasibility recovery
            tight_relaxation = True

            # -----------------------------------
            # Check if relaxation is tight
            # -----------------------------------
            tight_relaxation = True

            for od in self.net['OD_pairs']:
                d = self.demand[od]
                if d <= 1e-6:
                    continue

                opt_data = self.net['opt_data'][od]
                # Calculate the exact wait term from solver's auxiliary variables
                wait_term_val = sum((30.0 / s) * self.u[od, l, s].X for l in opt_data['gamma_odl'] for s in S_SET)
                walk_time = opt_data['walking_dist'] / self.W_SPEED
                C_od = self.args.vot1 * (opt_data['tau_pt'] + walk_time) + TRANSIT_FARE
                
                # Evaluate the slack of the inequality constraint
                solver_lhs = self.H_x[od].X + self.args.vot1 * wait_term_val + solver_x_pt[od] * (C_od + self.v_a[od])
                
                if solver_lhs < -0.01: # Numerical tolerance for tightness
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
                
                br_strategy = {
                    'frequencies': freq_opt, 
                    'fare': TRANSIT_FARE, 
                    'x_pt': solver_x_pt
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
                exact_x_pt = {}
                for od in self.net['OD_pairs']:
                    d = self.demand[od]
                    if d <= 1e-6:
                        exact_x_pt[od] = 0.0
                        continue
                        
                    opt_data = self.net['opt_data'][od]
                    valid_path = True
                    true_wait_time = 0
                    for l in opt_data['gamma_odl']:
                        if freq_opt[l] > 1e-6:
                            true_wait_time += 30.0 / freq_opt[l]
                        else:
                            valid_path = False; break
                    
                    if not valid_path:
                        exact_x_pt[od] = 0.0
                        continue
                    
                    walk_time = opt_data['walking_dist'] / self.W_SPEED
                    v_pt = -self.args.vot1 * (true_wait_time + opt_data['tau_pt'] + walk_time) - TRANSIT_FARE
                    exponent = self.v_a[od] - v_pt
                    
                    if exponent > 500: exact_x_pt[od] = 0.0
                    elif exponent < -500: exact_x_pt[od] = d
                    else: exact_x_pt[od] = d / (1.0 + np.exp(exponent))
                
                # Evaluate Repaired Upper Bound (Removes Tangent Error!)
                profit_repaired_ub = self.evaluate({
                    'frequencies': freq_opt, 
                    'fare': TRANSIT_FARE, 
                    'x_pt': exact_x_pt 
                })

                feasible_freq = freq_opt.copy()
                final_x_pt = {}
                sorted_S = sorted(list(S_SET))
                recovery_iterations = 0
            
                while True:
                    recovery_iterations += 1

                    # Exact non-linear logit demand under current frequencies
                    for od in self.net['OD_pairs']:
                        d = self.demand[od]
                        if d <= 1e-6:
                            final_x_pt[od] = 0.0
                            continue
                        
                        opt_data = self.net['opt_data'][od]
                        valid_path = True
                        true_wait_time = 0
                        for l in opt_data['gamma_odl']:
                            if feasible_freq[l] > 1e-6:
                                true_wait_time += 30.0/feasible_freq[l]
                            else:
                                valid_path = False
                                break
                        if not valid_path:
                            final_x_pt[od] = 0.0
                            continue
                        
                        walk_time = opt_data['walking_dist'] / self.W_SPEED
                        fixed_time = opt_data['tau_pt'] + walk_time
                        v_pt = -self.args.vot1 * (true_wait_time + fixed_time) - TRANSIT_FARE
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
                            ods_on_seg = [od for od in self.net['OD_pairs'] if (u_s, v_s, l) in self.net['opt_data'][od]['eta_e_od']]
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
                                ods_on_seg = [od for od in self.net['OD_pairs'] if (u_s, v_s, l) in self.net['opt_data'][od]['eta_e_od']]
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
                    'x_pt': final_x_pt
                }
                profit_lb = self.evaluate(br_strategy)

                # Print Final Gap Summary
                print(f"Repaired Logit Upper Bound:  {profit_repaired_ub:.2f}")
                print(f"Primal Lower Bound (True):   {profit_lb:.2f}")
                
                gap = ((profit_repaired_ub - profit_lb) / abs(profit_repaired_ub)) * 100 if profit_repaired_ub != 0 else 0
                print(f"Suboptimality Gap:           {gap:.2f}%\n")
                
                # Returns the tight_relaxation flag as requested!
                return profit_lb, br_strategy, False, gap

        else:
            print("Transit Best Response Failed to Solve.")
            return None, None, False, None
            
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