import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.constants import C_L_UNIT, S_SET, TRANSIT_FARE, MPH_TO_M_PER_MIN, AMOD_WAIT, BASE_LOAD_FACTOR

class TransitBestResponse:
    def __init__(self, args, net_data, demand_dict, amod_so_solution, transit_so_solution):

        self.args = args
        self.net = net_data
        self.demand = demand_dict
        self.transit_so = transit_so_solution

        self.W_SPEED = args.walk * MPH_TO_M_PER_MIN

        # Calculate fixed AMoD utility V_a based on Social Optimum path assignments
        self.v_a = self._calculate_amod_utility(amod_so_solution)

        self.m = gp.Model("Transit_BestResponse")
        self.m.params.MIPGap = args.mip
        self.m.params.NonConvex = 0

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

        # 3. Mirrored Geomspace PWL for Convex Entropy Term H(x) = x ln(x) - x ln(d-x)
        self.H_x = m.addVars(net['OD_pairs'], lb=-GRB.INFINITY, name="Entropy_H")
        for od in net['OD_pairs']:
            d = demand[od]
            if d > 1e-2:
                self.x_pt[od].ub = d - 1e-6
                half_pts = np.geomspace(1e-6, d / 2.0, num=25)
                right_pts = d - half_pts[::-1] 
                x_pts = np.unique(np.concatenate((half_pts, right_pts))).tolist()
                y_pts = [x * np.log(x) - x * np.log(d - x) for x in x_pts]
                m.addGenConstrPWL(self.x_pt[od], self.H_x[od], x_pts, y_pts)
                # for i in range(len(x_pts) - 1):
                #     slope = (y_pts[i+1] - y_pts[i]) / (x_pts[i+1] - x_pts[i])
                #     intercept = y_pts[i] - slope * x_pts[i]
                #     # This builds a purely continuous convex lower bound for H(x)
                #     m.addConstr(self.H_x[od] >= slope * self.x_pt[od] + intercept)
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
                m.addConstr(pass_on_seg <= (1 - BASE_LOAD_FACTOR) * C_L_UNIT * self.s_l[l])

        # 6. Objective: Maximize (Fare + Social Bonus) * ridership - Op Cost
        revenue_and_bonus = gp.quicksum((TRANSIT_FARE + self.args.omega) * self.x_pt[od] for od in net['OD_pairs'])
        op_cost = gp.quicksum(self.args.cl * self.s_l[l] for l in net['L'])
        m.setObjective(revenue_and_bonus - op_cost, GRB.MAXIMIZE)

    def solve(self):
            self.m.optimize()
            if self.m.status == GRB.OPTIMAL:
                br_strategy = {
                    'frequencies': {l: self.s_l[l].X for l in self.net['L']},
                    'fare': TRANSIT_FARE
                }
                return self.m.ObjVal, br_strategy
            else:
                print("Transit Best Response Failed to Solve.")
                return None, None
            
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