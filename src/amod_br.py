import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor


def obbt_worker(edges_chunk, net, demand, args, v_pt_fixed):
    """
    Worker function to build and solve OBBT for a chunk of edges.
    Runs on parallel CPU cores.
    """
    # 1. Initialize local model for this CPU worker
    m_obbt = gp.Model("OBBT_Worker")
    m_obbt.params.OutputFlag = 0
    
    # --- Algorithmic Speedups ---
    m_obbt.params.Method = 2       # Use Barrier Method (Fast for networks)
    m_obbt.params.Crossover = 0    # Skip crossover (Only need bounds, not routing)
    m_obbt.params.Threads = 1      # 1 thread per worker to prevent CPU choking

    # 2. Build Variables
    x_ap = m_obbt.addVars(net['paths_df']['path_id'].unique(), lb=0, name="x_path")
    x_a = m_obbt.addVars(net['OD_pairs'], lb=1e-6, ub=[demand[od] for od in net['OD_pairs']], name="x_a")
    r = m_obbt.addVars(net['E_a'], lb=0, name="r_ij") # Notice: No upper bound here!
    f = m_obbt.addVars(net['E_a'], lb=0, name="f_ij")

    # 3. Standard Flow Constraints
    for node in net['road_nodes']:
        rebal_in = r.sum('*', node)
        rebal_out = r.sum(node, '*')
        pass_out = gp.quicksum(x_a[o, d] for o, d in net['OD_pairs'] if net['zone_to_node'].get(o) == node)
        pass_in = gp.quicksum(x_a[o, d] for o, d in net['OD_pairs'] if net['zone_to_node'].get(d) == node)
        m_obbt.addConstr(rebal_in + pass_in == rebal_out + pass_out)

    for od in net['OD_pairs']:
        rel_paths = net['paths_df'][(net['paths_df']['o_zone'] == od[0]) & (net['paths_df']['d_zone'] == od[1])]['path_id'].unique()
        m_obbt.addConstr(x_a[od] == x_ap.sum(rel_paths))

    for e in net['E_a']:
        p_on_e = net['paths_df'][(net['paths_df']['u'] == e[0]) & (net['paths_df']['v'] == e[1])]['path_id'].unique()
        m_obbt.addConstr(f[e] == r[e] + x_ap.sum(p_on_e))

    # 4. Rigorous Cycle Breakers (Economic Limits)
    total_max_profit = 0
    total_max_revenue = 0

    for od in net['OD_pairs']:
        d = demand[od]
        if d < 1e-3: continue

        rel_paths = net['paths_df'][(net['paths_df']['o_zone'] == od[0]) & (net['paths_df']['d_zone'] == od[1])]
        if len(rel_paths) > 0:
            min_dist_km = min([sum(net['dist_map'][r_p.u, r_p.v] for r_p in net['paths_df'][net['paths_df']['path_id']==p].itertuples()) for p in rel_paths['path_id'].unique()]) / 1000.0
            c_min = args.ca * min_dist_km
        else:
            c_min = 0
        
        v_pt = v_pt_fixed[od]
        x_vals = np.linspace(1e-5, d-1e-5, 1000)

        revenue_curve = x_vals * (v_pt - np.log(x_vals) + np.log(d - x_vals))
        total_max_revenue += np.max(revenue_curve)

        profit_curve = revenue_curve - (c_min * x_vals)
        max_profit = np.max(profit_curve)

        # Only add to the budget if the OD pair is fundamentally profitable
        if max_profit > 0:
            total_max_profit += max_profit

    m_obbt.addConstr(gp.quicksum(args.ca * r[e] * (net['dist_map'][e] / 1000.0) for e in net['E_a']) <= total_max_profit)
    m_obbt.addConstr(gp.quicksum(args.ca * f[e] * (net['dist_map'][e] / 1000.0) for e in net['E_a']) <= total_max_revenue)

    # 5. Solve for the assigned chunk of edges
    r_ub_chunk = {}
    f_ub_chunk = {}
    
    for e in edges_chunk:
        m_obbt.setObjective(r[e], GRB.MAXIMIZE)
        m_obbt.optimize()
        r_ub_chunk[e] = m_obbt.ObjVal if m_obbt.status == GRB.OPTIMAL else 2 * net['Cap_map'][e]

        m_obbt.setObjective(f[e], GRB.MAXIMIZE)
        m_obbt.optimize()
        f_ub_chunk[e] = m_obbt.ObjVal if m_obbt.status == GRB.OPTIMAL else 3 * net['Cap_map'][e]
        
    return r_ub_chunk, f_ub_chunk

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

    def solve_upper_bound_relaxation(self):
        print("\nSolving covex relaxation with direct non-linear term bounding...")
        # Variables
        v = self.m.addVars(self.net['E_a'], lb=0, name="v_ij")
        v_T = self.m.addVars(self.net['E_a'], lb=0, name="v_T")

        # Constraints for defining v = f-r and the convex epigraph for f*T(b+f) using direct inequalities
        pwl_constrs = []
        for e in self.net['E_a']:
            b = self.net['b_map'][e]
            C = self.net['Cap_map'][e]
            T0 = self.net['T0_map'][e]

            p_on_e = self.net['paths_df'][(self.net['paths_df']['u'] == e[0]) & (self.net['paths_df']['v'] == e[1])]['path_id'].unique()
            c1 = self.m.addConstr(v[e] == self.x_ap.sum(p_on_e))
            pwl_constrs.append(c1)

            v_max = 2 * C
            v_pts = np.linspace(0, v_max, num=30).tolist()
            y_pts = [v_val * T0 * (1+self.args.alpha * ((b + v_val) / C) ** self.args.beta) for v_val in v_pts]

            # for i in range(len(v_pts) - 1):
            #     slope = (y_pts[i+1] - y_pts[i]) / (v_pts[i+1] - v_pts[i])
            #     intercept = y_pts[i] - slope * v_pts[i]
            #     c_lb = self.m.addConstr(v_T[e] >= slope * v[e] + intercept)
            #     pwl_constrs.append(c_lb)
            # Tangent lines for the relaxation
            for v_val in v_pts:
                ratio = (b + v_val) / C
                T_val = T0 * (1 + self.args.alpha * (ratio ** self.args.beta))
                y_val = v_val * T_val
                
                # Derivative of v * T(b+v)
                if b + v_val > 0:
                    grad = T0 * (1 + self.args.alpha * (ratio ** self.args.beta)) + \
                           v_val * T0 * (self.args.alpha * self.args.beta / C) * (ratio ** (self.args.beta - 1))
                else:
                    grad = T0
                    
                intercept = y_val - grad * v_val
                c_lb = self.m.addConstr(v_T[e] >= grad * v[e] + intercept)
                pwl_constrs.append(c_lb)
                
        # Objective
        road_time_convex = gp.quicksum(v_T[e] for e in self.net['E_a'])
        amod_op_cost = gp.quicksum(self.args.ca * self.f[e] * (self.net['dist_map'][e] / 1000) for e in self.net['E_a'])

        obj = (gp.quicksum(self.ent_obj[od] for od in self.net['OD_pairs'])
            - gp.quicksum(self.x_a[od] * self.v_pt_fixed[od] for od in self.net['OD_pairs'])
            + self.args.vot1 * road_time_convex
            + amod_op_cost)
        
        self.m.setObjective(obj, GRB.MINIMIZE)
        self.m.params.OutputFlag = 1

        self.m.optimize()
        
        if self.m.status == GRB.OPTIMAL:
            profit_ub = -self.m.ObjVal
        else:
            print(f"Relaxation failed with status {self.m.status}")
            profit_ub = None

        # Clean-up
        self.m.remove(pwl_constrs)
        self.m.remove([v[e] for e in self.net['E_a']])
        self.m.remove([v_T[e] for e in self.net['E_a']])
        self.m.params.OutputFlag = 0
        self.m.update()

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
            rebal_linear_surrogate = gp.quicksum(-T_k[e]*self.r[e] - (r_k[e]*grad_T_k[e])*self.f[e] + (r_k[e]*grad_T_k[e]*f_k[e]) for e in self.net['E_a'])
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
                
        exact_road_time = 0
        for e in self.net['E_a']:
            v_e = f_k[e] - r_k[e]
            ratio = (self.net['b_map'][e] + f_k[e]) / self.net['Cap_map'][e]
            T_e = self.net['T0_map'][e] * (1 + self.args.alpha * (ratio ** self.args.beta))
            exact_road_time += v_e * T_e
        exact_amod_op_cost = sum(self.args.ca * f_k[e] * (self.net['dist_map'][e] / 1000) for e in self.net['E_a'])
        exact_entropy_and_pt = 0
        for od in self.net['OD_pairs']:
            x_a_od = self.x_a[od].X
            d_od = self.demand[od]
            if x_a_od > 1e-6 and d_od > 1e-6:
                exact_entropy_and_pt += -x_a_od * self.v_pt_fixed[od] + (x_a_od * np.log(x_a_od) - x_a_od * np.log(d_od - x_a_od))
        true_negative_profit = exact_entropy_and_pt + self.args.vot1 * exact_road_time + exact_amod_op_cost
        true_profit_lb = -true_negative_profit
        return true_profit_lb, f_k, r_k
    
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
