import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

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
        self.f_T = m.addVars(net['E_a'], lb=0, name="f_T") # Auxiliary variable for strictly convex f * T(b+f)

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
                
                # m.addGenConstrPWL(self.x_a[od], self.ent_obj[od], x_pts, y_pts)
                # Replace addGenConstrPWL with continuous secant inequalities
                for i in range(len(x_pts) - 1):
                    slope = (y_pts[i+1] - y_pts[i]) / (x_pts[i+1] - x_pts[i])
                    intercept = y_pts[i] - slope * x_pts[i]
                    m.addConstr(self.ent_obj[od] >= slope * self.x_a[od] + intercept)
            else:
                m.addConstr(self.x_a[od] == 1e-6)
                m.addConstr(self.ent_obj[od] == 0)
        # Constraints
        # 1. Flow on road and convex epigraph for BPR
        for e in net['E_a']:
            p_on_e = net['paths_df'][(net['paths_df']['u'] == e[0]) & (net['paths_df']['v'] == e[1])]['path_id'].unique()
            m.addConstr(self.f[e] == self.r[e] + self.x_ap.sum(p_on_e))

            b = net['b_map'][e]
            C = net['Cap_map'][e]
            T0 = net['T0_map'][e]

            # Convex epigraph for f*T(b+f) using pre-computed points
            f_max = 2 * C
            f_pts = np.linspace(0, f_max, num=25).tolist()

            y_pts = []
            for f_val in f_pts:
                v_val = b + f_val
                ratio = v_val / C
                T_val = T0 * (1 + self.args.alpha * (ratio ** self.args.beta))
                y_pts.append(f_val * T_val)
            # Continuous secant inequalities for the single convex bowl
            for i in range(len(f_pts) - 1):
                slope = (y_pts[i+1] - y_pts[i]) / (f_pts[i+1] - f_pts[i])
                intercept = y_pts[i] - slope * f_pts[i]
                m.addConstr(self.f_T[e] >= slope * self.f[e] + intercept)

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

    def _compute_obbt_bounds(self):
        """
        Calculate strict upper bounds for rebalancing flow r_ij and total flow f_ij using Optimization-Based Bound Tightening (OBBT).
        Maximizes r_ij and f_ij on each edge subject only to network flow conservation and max demand.
        """
        print("\nRunning OBBT to find tight bounds for rebalancing and total flows on edges...")
        m_obbt = gp.Model("OBBT")
        m_obbt.params.OutputFlag = 0

        total_demand = sum(self.demand[od] for od in self.net['OD_pairs'])

        # Variables
        x_ap = m_obbt.addVars(self.net['paths_df']['path_id'].unique(), lb=0, name="x_path")
        x_a = m_obbt.addVars(self.net['OD_pairs'], lb=1e-6, ub = [self.demand[od] for od in self.net['OD_pairs']], name="x_a")
        r = m_obbt.addVars(self.net['E_a'], lb=0, name="r_ij")
        f = m_obbt.addVars(self.net['E_a'], lb=0, name="f_ij")

        # Constraints
        # Flow conservation
        for node in self.net['road_nodes']:
            rebal_in = r.sum('*', node)
            rebal_out = r.sum(node, '*')
            pass_out = gp.quicksum(x_a[o, d] for o, d in self.net['OD_pairs'] if self.net['zone_to_node'].get(o) == node)
            pass_in = gp.quicksum(x_a[o, d] for o, d in self.net['OD_pairs'] if self.net['zone_to_node'].get(d) == node)
            m_obbt.addConstr(rebal_in + pass_in == rebal_out + pass_out)

        # Path definition
        for od in self.net['OD_pairs']:
            rel_paths = self.net['paths_df'][(self.net['paths_df']['o_zone'] == od[0]) & (self.net['paths_df']['d_zone'] == od[1])]['path_id'].unique()
            m_obbt.addConstr(x_a[od] == x_ap.sum(rel_paths))

        for e in self.net['E_a']:
            p_on_e = self.net['paths_df'][(self.net['paths_df']['u'] == e[0]) & (self.net['paths_df']['v'] == e[1])]['path_id'].unique()
            m_obbt.addConstr(f[e] == r[e] + x_ap.sum(p_on_e))

        total_max_profit = 0
        total_max_revenue = 0

        for od in self.net['OD_pairs']:
            d = self.demand[od]
            if d < 1e-3: continue

            # Find the minumum physical cost to serve one passenger for this OD pair
            rel_paths = self.net['paths_df'][(self.net['paths_df']['o_zone'] == od[0]) & (self.net['paths_df']['d_zone'] == od[1])]
            if len(rel_paths) > 0:
                min_dist_km = min([sum(self.net['dist_map'][r.u, r.v] for r in self.net['paths_df'][self.net['paths_df']['path_id']==p].itertuples()) for p in rel_paths['path_id'].unique()]) / 1000.0
                c_min = self.args.ca * min_dist_km
            else:
                c_min = 0
            
            v_pt = self.v_pt_fixed[od]
            x_vals = np.linspace(1e-5, d-1e-5, 1000)

            # Calculate revenue curve
            revenue_curve = x_vals * (v_pt - np.log(x_vals) + np.log(d - x_vals))
            total_max_revenue += np.max(revenue_curve)

            # Calculate profit curve
            profit_curve = revenue_curve - c_min * x_vals
            max_profit = np.max(profit_curve)

            if max_profit > 0:
                total_max_profit += max_profit

        # Constraint A: rebalnacing cost cannot exceed total maximum profit
        m_obbt.addConstr(gp.quicksum(self.args.ca * r[e] * (self.net['dist_map'][e] / 1000.0) for e in self.net['E_a']) <= total_max_profit, name="max_profit_rebal_limit")

        # Constraint B: total cost cannot exceed total maximum revenue
        m_obbt.addConstr(gp.quicksum(self.args.ca * f[e] * (self.net['dist_map'][e] / 1000.0) for e in self.net['E_a']) <= total_max_revenue, name="max_revenue_total_cost_limit")
        
        r_ub = {}
        f_ub = {}
        # Maximize r_e and f_e for each edge
        for e in tqdm(self.net['E_a'], desc="OBBT Progress"):
            # Maximize rebalancing flow r_e
            m_obbt.setObjective(r[e], GRB.MAXIMIZE)
            m_obbt.optimize()
            if m_obbt.status == GRB.OPTIMAL:
                r_ub[e] = r[e].X
            else:
                r_ub[e] = 2 * self.net['Cap_map'][e] # Fall back to a large number
                print(f"Edge {e}: OBBT failed to find upper bound, status code {m_obbt.status}. Fall back to 2*capacity = {r_ub[e]}")
            
            # Maximize total AMoD flow f_e
            m_obbt.setObjective(f[e], GRB.MAXIMIZE)
            m_obbt.optimize()
            if m_obbt.status == GRB.OPTIMAL:
                f_ub[e] = f[e].X
            else:
                f_ub[e] = r_ub[e] + total_demand # Fallack
                print(f"Edge {e}: OBBT failed to find upper bound for total flow, status code {m_obbt.status}. Fall back to r_ub + total_demand = {f_ub[e]}")
        return r_ub, f_ub
    
    def solve_upper_bound_relaxation(self):
        """
        Solve AMoD best response using McCormick envelopes.
        Returns an upper bound on the maximum profit.
        """
        # Step 1: get tight upper bounds for r_ij and f_ij using OBBT
        r_ub, f_ub = self._compute_obbt_bounds()
        print("OBBT finished. Applying bounds to get convex relaxation with McCormick envelopes...")

        # Step 2: Add variables for the bilinear terms in the objective
        T_e = self.m.addVars(self.net['E_a'], name="T_e")
        w = self.m.addVars(self.net['E_a'], name="w_e")

        # Track added constraints for cleanup later
        mccormick_constrs = []
        pwl_constrs = []

        # Step 3: Define T_e bounds and constraints
        T_ub = {}
        T_lb = {}
        for e in self.net['E_a']:
            b = self.net['b_map'][e]
            C = self.net['Cap_map'][e]
            T0 = self.net['T0_map'][e]

            # Lower bound on T_e: only background traffic 
            T_lb[e] = T0 * (1 + self.args.alpha * ((b / C) ** self.args.beta))
            # Upper bound on T_e: background + max possible flow
            v_max = b + f_ub[e]
            T_ub[e] = T0 * (1 + self.args.alpha * ((v_max / C) ** self.args.beta))

            # PWL approximation for T_e based on f_e
            f_pts = np.linspace(0, f_ub[e] + 1e-3, num=25).tolist()
            T_pts = [T0 * (1 + self.args.alpha * (((b + f_val) / C) ** self.args.beta)) for f_val in f_pts]
            gc = self.m.addGenConstrPWL(self.f[e], T_e[e], f_pts, T_pts)
            pwl_constrs.append(gc)

            # McCormick Envelope Constraints
            # Relaxes the bilinear product w_e ≈ r_e * T_e
            c1 = self.m.addConstr(w[e] >= self.r[e] * T_lb[e])
            c2 = self.m.addConstr(w[e] >= r_ub[e] * T_e[e] + self.r[e] * T_ub[e] - r_ub[e] * T_ub[e])
            c3 = self.m.addConstr(w[e] <= self.r[e] * T_ub[e])
            c4 = self.m.addConstr(w[e] <= r_ub[e] * T_e[e] + self.r[e] * T_lb[e] - r_ub[e] * T_lb[e])
            mccormick_constrs.extend([c1, c2, c3, c4])

        # Objective
        road_time_convex = gp.quicksum(self.f_T[e] for e in self.net['E_a'])
        rebal_mccormick = gp.quicksum(-w[e] for e in self.net['E_a'])
        amod_op_cost = gp.quicksum(self.args.ca * self.f[e] * (self.net['dist_map'][e] / 1000) for e in self.net['E_a'])

        obj = (gp.quicksum(self.ent_obj[od] for od in self.net['OD_pairs'])
               - gp.quicksum(self.x_a[od] * self.v_pt_fixed[od] for od in self.net['OD_pairs'])
               + self.args.vot1 * (road_time_convex + rebal_mccormick)
               + amod_op_cost)

        self.m.setObjective(obj, GRB.MINIMIZE)

        print("\nSolving Convex Relaxation (Upper Bound)...")
        self.m.params.OutputFlag = 1
        self.m.optimize()

        if self.m.status == GRB.OPTIMAL:
            profit_ub = -self.m.ObjVal
        else:
            print(f"Relaxation failed with status {self.m.status}")
            profit_ub = None
        
        # Clean-up: Remove all relaxation-specific variables and constraints
        self.m.remove(mccormick_constrs)
        self.m.remove(pwl_constrs)
        self.m.remove([T_e[e] for e in self.net['E_a']])
        self.m.remove([w[e] for e in self.net['E_a']])
        self.m.update() # Apply the removals to the model state

        return profit_ub

    def solve_sca(self, max_iter=20, tol=1e-3, trust_region_delta_r=50, trust_region_delta_f=50, f_initial=None, r_initial=None, x_a_initial=None, x_ap_initial=None):
        """
        Solves the AMoD Best Response using Successive Convex Approximation.
        Returns a LOWER BOUND on the maximum profit.
        """

        # ### --- TEMPORARY PLOTTING BLOCK 1: Imports & Setup --- ###
        if self.args.plot:    
            plt.ion() # Turn on interactive mode
            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax2 = ax1.twinx()

        iter_history = []
        obj_history = []
        diff_history = []
        # ### --------------------------------------------------- ###

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

        # ### --- TEMPORARY PLOTTING BLOCK 2: Wrap range in tqdm --- ###
        for iteration in tqdm(range(max_iter), desc="SCA Iterations"):
        # ### ------------------------------------------------------ ###
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
            road_time_convex = gp.quicksum(self.f_T[e] for e in self.net['E_a'])
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
            if self.m.status != GRB.OPTIMAL:
                print(f"SCA Iteration {iteration}: Subproblem not optimal, status code {self.m.status}")
                break
            
            # Step 3: Check Convergence
            max_diff = 0
            for e in self.net['E_a']:
                diff = abs(self.f[e].X - f_k[e])
                if diff > max_diff: max_diff = diff
                f_k[e] = self.f[e].X
                r_k[e] = self.r[e].X
                
            # ### --- TEMPORARY PLOTTING BLOCK 3: Dynamic Plotting --- ###
            iter_history.append(iteration)
            obj_history.append(-self.m.ObjVal)
            diff_history.append(max_diff)

            if self.args.plot:
                fig.clf()

                ax1 = fig.add_subplot(111)
                ax2 = ax1.twinx()

                ax1.set_xlabel('SCA Iteration')
                ax1.set_ylabel('Surrogate Profit Lower Bound', color='tab:blue')
                ax1.plot(iter_history, obj_history, color='tab:blue', marker='o', label='Profit')
                ax1.tick_params(axis='y', labelcolor='tab:blue')

                ax2.set_ylabel('Max Flow Change', color='tab:red')
                ax2.plot(iter_history, diff_history, color='tab:red', marker='x', linestyle='--', label='Flow Diff')
                ax2.tick_params(axis='y', labelcolor='tab:red')

                plt.title(f"SCA Convergence Tracking")
                fig.tight_layout()
                plt.draw()
                plt.pause(0.01) # Small pause to allow GUI to update
            # ### ---------------------------------------------------- ###

            tqdm.write(f"SCA Iteration {iteration}: Max flow change = {max_diff:.4f}")
            if max_diff <= tol:
                tqdm.write("SCA Converged!")
                break
                
        # ### --- TEMPORARY PLOTTING BLOCK 4: Keep Plot Open --- ###
        if self.args.plot:
            plt.ioff()
            plt.close()
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx()
        ax1.set_xlabel('SCA Iteration')
        ax1.set_ylabel('Surrogate Profit Lower Bound', color='tab:blue')
        ax1.plot(iter_history, obj_history, color='tab:blue', marker='o', label='Profit')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax2.set_ylabel('Max Flow Change', color='tab:red')
        ax2.plot(iter_history, diff_history, color='tab:red', marker='x', linestyle='--', label='Flow Diff')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        plt.title(f"SCA Convergence Tracking_{self.args.bl}_{self.args.ca}_{self.args.cl}")
        fig.tight_layout()

        path = os.path.join(self.args.output_dir, "images")
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(os.path.join(self.args.output_dir, "images", f"sca_convergence_{self.args.bl}_{self.args.ca}_{self.args.cl}.png"), bbox_inches='tight') # Save the final plot as a file
        plt.close(fig)
        # plt.show() # Uncomment this if you want the program to pause and wait for you to close the plot
        # ### -------------------------------------------------- ###
                
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
