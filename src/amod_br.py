import gurobipy as gp
from gurobipy import GRB
import numpy as np

class AMoD_BestResponse:
    """
    Algorithm to calculate the solution of AMoD best response by calculating the upper and lower bound through relaxation and SCA.
    """
    def __init__(self, args, network_data, demand_dict, fixed_transit_solution):
        """
        network_data: dict containing E_a, T0_map, Cap_map, b_map, dist_map, OD_pairs, paths_df, etc.
        fixed_transit_solution: dict containing {line: frequency} and transit passenger utilities from Step 1.
        """
        self.args = args
        self.net = network_data
        self.demand = demand_dict

        # Pre-calculate fixed transit utility for each OD pair
        # V_pt = VOT_USER * (t_in_veh+t_wait+t)walk) + Fare
        self.v_pt_fixed = self._calculate_fixed_transit_utility(fixed_transit_solution)

        # Build base model
        self.m = gp.Model("AMoD_Best_Response")
        self.m.params.NonConvex = 0
        self.m.params.OutputFlag = 0
        self._build_base_model()
    
    def _calculate_fixed_transit_utility(self, transit_solution):
        v_pt = {}
        for od in self.net['OD_pairs']:
            opt_data = self.net['opt_data'][od]
            # Wait time based on fixed frequencies from social optimum
            wait_time = sum(0.5 * 60 / transit_solution['frequencies'][l] for l in opt_data['gamma_odl'] if transit_solution['frequencies'][l] > 0)
            walk_time = opt_data['walking_dist'] / (self.args.walk * 1000 / 60) # convert to minutes
            in_veh_time = opt_data['tau_pt']

            v_pt[od] = self.args.vot1 * (in_veh_time + walk_time + wait_time) + transit_solution['fare']
        
        return v_pt

    def _build_base_model(self):
        """Build variables and constraints that remain constant across all SCA iterations"""
        m = self.m
        net = self.net

        # Variables
        self.x_ap = m.addVars(net['paths_df']['path_id'].unique(), lb=0, name="x_path") # Flow on AMoD path
        self.x_a = m.addVars(net['OD_pairs'], lb=1e-6, name="x_a")
        self.x_pt = m.addVars(net['OD_pairs'], lb=1e-6, name="x_pt") # Flow on transit route

        self.r = m.addVars(net['E_a'], lb=0, name="r_ij") # Rebalancing flow
        self.f = m.addVars(net['E_a'], lb=0, name="f_ij") # AMOD flow
        self.v = m.addVars(net['E_a'], lb=[net['b_map'][e] for e in net['E_a']], name="v_total") # Total volume (b + f)
        self.v_ratio = m.addVars(net['E_a'], lb=0, name="v_ratio")
        self.v_ratio_pow = m.addVars(net['E_a'], lb=0, name="v_ratio_pow") # Auxiliary for v^(beta+1) in BPR
        self.v_ratio_beta = self.m.addVars(self.net['E_a'], lb=0, name="v_ratio_beta") # Auxiliary for v^beta in BPR

        # PWL Auxiliary for entropy term: y = x*ln(x) - x*ln(d-x)
        self.ent_obj = m.addVars(net['OD_pairs'], lb=-GRB.INFINITY, name="entropy_term")
        for od in net['OD_pairs']:
            d = self.demand[od]
            if d > 1e-2:
                self.x_a[od].ub = d - 1e-6
                # 1. Create a geomspace from the left bound to the exact middle
                half_pts = np.geomspace(1e-6, d / 2.0, num=25)
                
                # 2. Mirror it to capture the right boundary's curvature
                # half_pts[::-1] reverses the array, so we subtract from d
                right_pts = d - half_pts[::-1] 
                
                # 3. Combine them and use np.unique to sort and remove the duplicated d/2 point
                x_pts = np.unique(np.concatenate((half_pts, right_pts))).tolist()
                
                # 4. Calculate y values
                y_pts = [x * np.log(x) - x * np.log(d - x) for x in x_pts]
                
                m.addGenConstrPWL(self.x_a[od], self.ent_obj[od], x_pts, y_pts)
            else:
                m.addConstr(self.x_a[od] == 1e-6)
                m.addConstr(self.ent_obj[od] == 0)
        # Constraints
        # 1. Flow on road and BPR
        for e in net['E_a']:
            p_on_e = net['paths_df'][(net['paths_df']['u'] == e[0]) & (net['paths_df']['v'] == e[1])]['path_id'].unique()
            m.addConstr(self.f[e] == self.r[e] + self.x_ap.sum(p_on_e))
            m.addConstr(self.v[e] == net['b_map'][e] + self.f[e])
            m.addConstr(self.v_ratio[e] == self.v[e] / net['Cap_map'][e])
            m.addGenConstrPow(self.v_ratio[e], self.v_ratio_pow[e], self.args.beta + 1)
            m.addGenConstrPow(self.v_ratio[e], self.v_ratio_beta[e], self.args.beta)

        # 2. Demand split
        for od in net['OD_pairs']:
            m.addConstr(self.x_a[od] + self.x_pt[od] == self.demand[od])
            rel_paths = net['paths_df'][(net['paths_df']['o_zone'] == od[0]) & (net['paths_df']['d_zone'] == od[1])]['path_id'].unique()
            m.addConstr(self.x_a[od] == self.x_ap.sum(rel_paths))
        
        # 3. Flow conservation for rebalancing
        for node in net['road_nodes']:
            # Vehicles arriving/departing node by rebalancing
            rebal_in = self.r.sum('*', node)
            rebal_out = self.r.sum(node, '*')
            
            # Vehicles consumed by departing passengers at this node
            pass_out = gp.quicksum(self.x_a[o, d] for o, d in net['OD_pairs'] if net['zone_to_node'].get(o) == node)
            # Vehicles produced by arriving passengers at this node
            pass_in = gp.quicksum(self.x_a[o, d] for o, d in net['OD_pairs'] if net['zone_to_node'].get(d) == node)
            
            # Equation: Vehicles In + Pass In = Vehicles Out + Pass Out
            m.addConstr(rebal_in + pass_in == rebal_out + pass_out, name=f"flow_bal_{node}")

    def solve_sca(self, max_iter=20, tol=1e-3, trust_region_delta_r=50, trust_region_delta_f=50, f_initial=None, r_initial=None, x_a_initial=None, x_ap_initial=None):
        """
        Solves the AMoD Best Response using Successive Convex Approximation.
        Returns a LOWER BOUND on the maximum profit.
        """
        # Step 0: initialize flows (using step 1 social optimum or any feasible solution)
        if f_initial is None or r_initial is None:
            f_k = {e: 0 for e in self.net['E_a']}
            r_k = {e: 0 for e in self.net['E_a']}
        else:
            f_k = f_initial.copy()
            r_k = r_initial.copy()

        # ---------------------------
        # Warm start with so solution
        # ---------------------------
        if x_a_initial is not None:
            for od in self.net['OD_pairs']:
                self.x_a[od].Start = x_a_initial[od]
        if x_ap_initial is not None:
            for p in self.x_ap.keys():
                self.x_ap[p].Start = x_ap_initial.get(p, 0.0)
        if f_initial is not None:
            for e in self.net['E_a']:
                self.f[e].Start = f_initial[e]
                self.r[e].Start = r_initial[e]
        
        # Trust region constraints
        tr_constrs = {}

        for iteration in range(max_iter):
            # Step 1: Calculate taylor expansion
            T_k = {}
            grad_T_k = {}
            for e in self.net['E_a']:
                v_k = self.net['b_map'][e] + f_k[e]
                ratio = v_k / self.net['Cap_map'][e]

                # T(v) = T0 * (1 + alpha * (v/c)^beta)
                T_k[e] = self.net['T0_map'][e] * (1 + self.args.alpha * (ratio**self.args.beta))
                # T'(v) = T0 * alpha * (beta/C) * (v/C)^{beta-1))
                if v_k > 0:
                    grad_T_k[e] = self.net['T0_map'][e] * self.args.alpha * (self.args.beta / self.net['Cap_map'][e]) * (ratio**(self.args.beta - 1))
                else:
                    grad_T_k[e] = 0

            # Step 2: Set strictly convex objective
            road_time_convex = gp.quicksum(
                # v_e * T(v_e)
                self.net['T0_map'][e] * (self.v[e] + self.args.alpha * self.net['Cap_map'][e] * self.v_ratio_pow[e])
                # - b_e * T(v_e)
                - self.net['b_map'][e] * self.net['T0_map'][e] * (1 + self.args.alpha * self.v_ratio_beta[e])
                for e in self.net['E_a']
            )
            rebal_linear_surrogate = gp.quicksum(-T_k[e]*self.r[e] - (r_k[e]*grad_T_k[e])*self.f[e] for e in self.net['E_a'])
            amod_op_cost = gp.quicksum(self.args.ca * self.f[e] * (self.net['dist_map'][e] / 1000) for e in self.net['E_a'])
            
            obj = (gp.quicksum(self.ent_obj[od] for od in self.net['OD_pairs'])
                   - gp.quicksum(self.x_a[od] * self.v_pt_fixed[od] for od in self.net['OD_pairs'])
                   + self.args.vot1 * (road_time_convex + rebal_linear_surrogate)
                   + amod_op_cost)

            self.m.setObjective(obj, GRB.MINIMIZE)
        
            # Update Trust Region
            if tr_constrs:
                self.m.remove(list(tr_constrs.values()))
            tr_constrs = {}
            for e in self.net['E_a']:
                tr_constrs[f"tr_f_up_{e}"] = self.m.addConstr(self.f[e] <= f_k[e] + trust_region_delta_f)
                tr_constrs[f"tr_f_dn_{e}"] = self.m.addConstr(self.f[e] >= max(0, f_k[e] - trust_region_delta_f))
                tr_constrs[f"tr_r_up_{e}"] = self.m.addConstr(self.r[e] <= r_k[e] + trust_region_delta_r)
                tr_constrs[f"tr_r_dn_{e}"] = self.m.addConstr(self.r[e] >= max(0, r_k[e] - trust_region_delta_r))
                
            # Solve the convex subproblem
            self.m.optimize()
            
            # Step 3: Check Convergence
            max_diff = 0
            for e in self.net['E_a']:
                diff = abs(self.f[e].X - f_k[e])
                if diff > max_diff: max_diff = diff
                f_k[e] = self.f[e].X
                r_k[e] = self.r[e].X
                
            print(f"SCA Iteration {iteration}: Max flow change = {max_diff:.4f}")
            if max_diff <= tol:
                print("SCA Converged!")
                break
                
        # Return negative objective (which is Maximum Profit) and the flows
        return -self.m.ObjVal, f_k, r_k
    
    def evaluate(self, amod_solution):
        """
        Calculate AMoD utility (objective) for a given solution. The solution must contain:
        - 'x_a': dict of AMoD passenger flow per OD pair
        - 'pi_a': dict of AMoD price per OD pair
        - 'f': dict of AMoD vehicle flow per edge        
        """
        revenue = sum(amod_solution['pi_a'][od] * amod_solution['x_a'][od] for od in self.net['OD_pairs'])
        op_cost = sum(self.args.ca * amod_solution['f'][e] * (self.net['dist_map'][e] / 1000.0) for e in self.net['E_a'])
        
        profit = revenue - op_cost
        return profit     
