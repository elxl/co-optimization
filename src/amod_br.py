import gurobipy as gp
import pandas as pd
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import multiprocessing as mp
from src.constants import AMOD_WAIT, MPH_TO_M_PER_MIN
from src.transit_paths import choose_path


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
        self.W_SPEED = args.walk * MPH_TO_M_PER_MIN

        # Pre-calculate fixed transit utility for each OD pair
        # V_pt = -VOT_USER * (t_in_veh+t_wait+t)walk) - Fare
        self.v_pt_fixed = self._calculate_fixed_transit_utility(fixed_transit_solution)

        # Build base model
        self.m = gp.Model("AMoD_Best_Response")
        self.m.params.NonConvex = 0
        self.m.params.OutputFlag = 0
        self._build_base_model()
    
    def _calculate_fixed_transit_utility(self, transit_solution):
        v_pt = {}
        self.pt_path_choice = {}
        freqs = transit_solution['frequencies']
        prefer = transit_solution.get('path_choice', {})
        for od in self.net['OD_pairs']:
            paths = self.net['pt_paths'][od]
            # Passengers take the min-time candidate path under the fixed frequencies,
            # keeping the transit player's own choice on ties.
            k, t_pt = choose_path(paths, freqs, self.W_SPEED, prefer=prefer.get(od))
            if t_pt == float('inf'):
                # Every candidate uses an unserved line: fall back to skipping those lines
                # in the waiting time (the GTFS zero-frequency convention).
                path = paths[k]
                wait_time = sum(0.5 * 60 / freqs[l] for l in path['gamma_odl'] if freqs[l] > 0)
                t_pt = path['tau_pt'] + path['walking_dist'] / self.W_SPEED + wait_time

            self.pt_path_choice[od] = k
            v_pt[od] = -self.args.vot1 * t_pt - transit_solution['fare']

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
                half_pts = np.geomspace(1e-6, d / 2.0, num=20)
                
                # 2. Mirror it to capture the right boundary's curvature
                # half_pts[::-1] reverses the array, so we subtract from d
                right_pts = d - half_pts[::-1] 
                
                # 3. Combine them and use np.unique to sort and remove the duplicated d/2 point
                x_pts = np.unique(np.concatenate((half_pts, right_pts))).tolist()
                
                # m.addGenConstrPWL(self.x_a[od], self.ent_obj[od], x_pts, y_pts)
                # Replace secants with continuous tangent inequalities (underestimators)
                for x_val in x_pts:
                    # y = x * ln(x) - x * ln(d - x)
                    y_val = x_val * np.log(x_val) - x_val * np.log(d - x_val)
                    
                    # Derivative
                    grad = np.log(x_val) + 1 - np.log(d - x_val) + (x_val / (d - x_val))
                    
                    # Tangent line: y - y_val = grad * (x - x_val) => y = grad * x + intercept
                    intercept = y_val - grad * x_val
                    
                    m.addConstr(self.ent_obj[od] >= grad * self.x_a[od] + intercept)
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

            # Continuous tangent inequalities (underestimators) for the convex bowl
            for f_val in f_pts:
                ratio = (b + f_val) / C
                T_val = T0 * (1 + self.args.alpha * (ratio ** self.args.beta))
                y_val = f_val * T_val
                
                # Derivative of f * T(b+f) using the product rule
                if b + f_val > 0:
                    grad = T0 * (1 + self.args.alpha * (ratio ** self.args.beta)) + \
                           f_val * T0 * (self.args.alpha * self.args.beta / C) * (ratio ** (self.args.beta - 1))
                else:
                    grad = T0
                    
                # Tangent line: y - y_val = grad * (f - f_val) => y = grad * f + intercept
                intercept = y_val - grad * f_val
                
                m.addConstr(self.f_T[e] >= grad * self.f[e] + intercept)

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
            v_pts = np.linspace(0, v_max, num=25).tolist()
            y_pts = [v_val * T0 * (1+self.args.alpha * ((b + v_val) / C) ** self.args.beta) for v_val in v_pts]


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
            + gp.quicksum(self.x_a[od] * self.v_pt_fixed[od] for od in self.net['OD_pairs'])
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

    def solve_sca(self, max_iter=50, tol=1e-3, trust_region_delta_r=50, trust_region_delta_f=50, f_initial=None, r_initial=None, x_a_initial=None, x_ap_initial=None, min_delta=1e-4):
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
        true_obj_history = []
        diff_history = []
        accepted_steps = 0
        # ### --------------------------------------------------- ###

        # Step 0: initialize flows (using step 1 social optimum or any feasible solution)
        if f_initial is None or r_initial is None:
            f_k = {e: 0 for e in self.net['E_a']}
            r_k = {e: 0 for e in self.net['E_a']}
            x_a_k = {od: 0 for od in self.net['OD_pairs']}
            x_ap_k = {p: 0 for p in self.x_ap.keys()}
        else:
            f_k = f_initial.copy()
            r_k = r_initial.copy()
            x_a_k = {od: 0 for od in self.net['OD_pairs']}
            x_ap_k = {p: 0 for p in self.x_ap.keys()}

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

        # Dynamic Trust Region & Helper Function ---
        delta_f = trust_region_delta_f
        delta_r = trust_region_delta_r
        current_true_profit = -float('inf')

        def calc_true_profit(f_dict, r_dict, x_a_dict):
            exact_road_time = 0
            for e in self.net['E_a']:
                v_e = f_dict[e] - r_dict[e]
                ratio = (self.net['b_map'][e] + f_dict[e]) / self.net['Cap_map'][e]
                T_e = self.net['T0_map'][e] * (1 + self.args.alpha * (ratio ** self.args.beta))
                exact_road_time += v_e * T_e
            
            exact_amod_op_cost = sum(self.args.ca * f_dict[e] * (self.net['dist_map'][e] / 1000) for e in self.net['E_a'])
            
            exact_entropy_and_pt = 0
            for od in self.net['OD_pairs']:
                x_a_od = x_a_dict[od]
                d_od = self.demand[od]
                if x_a_od > 1e-6 and d_od > 1e-6:
                    exact_entropy_and_pt += x_a_od * self.v_pt_fixed[od] + (x_a_od * np.log(x_a_od) - x_a_od * np.log(d_od - x_a_od))
            
            return -(exact_entropy_and_pt + self.args.vot1 * exact_road_time + exact_amod_op_cost)
        
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
                   + gp.quicksum(self.x_a[od] * self.v_pt_fixed[od] for od in self.net['OD_pairs'])
                   + self.args.vot1 * (road_time_convex + rebal_linear_surrogate)
                   + amod_op_cost)

            self.m.setObjective(obj, GRB.MINIMIZE)
        
            # Update Trust Region
            if tr_constrs:
                self.m.remove(list(tr_constrs.values()))
            tr_constrs = {}
            for e in self.net['E_a']:
                tr_constrs[f"tr_f_up_{e}"] = self.m.addConstr(self.f[e] <= f_k[e] + delta_f)
                tr_constrs[f"tr_f_dn_{e}"] = self.m.addConstr(self.f[e] >= max(0, f_k[e] - delta_f))
                tr_constrs[f"tr_r_up_{e}"] = self.m.addConstr(self.r[e] <= r_k[e] + delta_r)
                tr_constrs[f"tr_r_dn_{e}"] = self.m.addConstr(self.r[e] >= max(0, r_k[e] - delta_r))
                
            # Solve the convex subproblem
            self.m.optimize()
            if self.m.status != GRB.OPTIMAL:
                print(f"SCA Iteration {iteration}: Subproblem not optimal, status code {self.m.status}")
                break

            # Evaluate Candidate Step
            f_cand = {e: self.f[e].X for e in self.net['E_a']}
            r_cand = {e: self.r[e].X for e in self.net['E_a']}
            x_a_cand = {od: self.x_a[od].X for od in self.net['OD_pairs']}
            
            cand_true_profit = calc_true_profit(f_cand, r_cand, x_a_cand)

            if cand_true_profit >= current_true_profit - 1e-4: # 1e-4 tolerance for numerical solver drift
                # ACCEPT STEP
                accepted_steps += 1
                max_diff = max(abs(f_cand[e] - f_k[e]) for e in self.net['E_a'])
                
                f_k = f_cand.copy()
                r_k = r_cand.copy()
                x_a_k = x_a_cand.copy()
                x_ap_k = {p: self.x_ap[p].X for p in self.x_ap.keys()}
                current_true_profit = cand_true_profit
                
                tqdm.write(f"SCA Iteration {iteration}: Accepted. Max flow change = {max_diff:.4f}")

                iter_history.append(accepted_steps)
                obj_history.append(-self.m.ObjVal)
                true_obj_history.append(current_true_profit)
                diff_history.append(max_diff)
                
                if max_diff <= tol:
                    tqdm.write("SCA Converged!")
                    break
            else:
                # REJECT STEP
                tqdm.write(f"SCA Iteration {iteration}: Rejected. True profit dropped. Shrinking TR.")
                delta_f *= 0.5
                delta_r *= 0.5
                max_diff = float('inf') # Prevent convergence trigger on a rejected step    

                if delta_f < min_delta and delta_r < min_delta:
                    tqdm.write(f"SCA Terminated: Trust region collapsed below {min_delta}. Reached local optimum.")
                    break
                
            # ### --- TEMPORARY PLOTTING BLOCK 3: Dynamic Plotting --- ###

            if self.args.plot:
                fig.clf()

                ax1 = fig.add_subplot(111)
                ax2 = ax1.twinx()

                ax1.set_xlabel('SCA Iteration')
                ax1.set_ylabel('True objective', color='tab:blue')
                ax1.plot(iter_history, true_obj_history, color='tab:blue', marker='o', label='Profit')
                ax1.tick_params(axis='y', labelcolor='tab:blue')

                ax2.set_ylabel('Max Flow Change', color='tab:red')
                ax2.plot(iter_history, diff_history, color='tab:red', marker='x', linestyle='--', label='Flow Diff')
                ax2.tick_params(axis='y', labelcolor='tab:red')

                plt.title(f"SCA Convergence Tracking")
                fig.tight_layout()
                plt.draw()
                plt.pause(0.01) # Small pause to allow GUI to update
            # ### ---------------------------------------------------- ###
                
        # ### --- TEMPORARY PLOTTING BLOCK 4: Keep Plot Open --- ###
        if self.args.plot:
            plt.ioff()
            plt.close()
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx()
        ax1.set_xlabel('SCA Iteration')
        ax1.set_ylabel('True objective', color='tab:blue')
        ax1.plot(iter_history, true_obj_history, color='tab:blue', marker='o', label='Profit')
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
        
        true_profit_lb = calc_true_profit(f_k, r_k, {od: self.x_a[od].X for od in self.net['OD_pairs']})

        # Summarize convergence results
        sca_converge = {
            'iterations': iter_history,
            'obj_profit': obj_history,
            'true_profit': true_obj_history,
            'max_flow_diff': diff_history
        }
        return true_profit_lb, f_k, r_k, x_a_k, x_ap_k, sca_converge
    
    def get_results_dataframe(self, amod_res, transit_solution):
        """
        Recover AMoD prices and return OD-level statistics for the AMoD Best Response.
        amod_res: dict containing 'f', 'x_a', 'x_ap' from the accepted SCA state.
        transit_solution: dict containing 'frequencies' from the fixed transit strategy.
        """
        congested_time_map = {}
        for e in self.net['E_a']:
            v_total = self.net['b_map'][e] + amod_res['f'][e]
            ratio = v_total / self.net['Cap_map'][e]
            congested_time_map[e] = self.net['T0_map'][e] * (1 + self.args.alpha * (ratio**self.args.beta))

        line_wait_map = {l: (0.5 * 60 / transit_solution['frequencies'][l]) if transit_solution['frequencies'][l] > 0 else 0 for l in self.net['L']}

        detailed_summary = []
        for od in self.net['OD_pairs']:
            d = self.demand[od]
            rel_paths = self.net['paths_df'][(self.net['paths_df']['o_zone'] == od[0]) & (self.net['paths_df']['d_zone'] == od[1])]['path_id'].unique()
            
            amod_flow = amod_res['x_a'][od]
            transit_flow = d - amod_flow

            best_p = sorted(rel_paths)[0] if len(rel_paths) > 0 else None

            if amod_flow > 1e-6:
                pot_amod_dist = sum(amod_res['x_ap'].get(p, 0) * sum(self.net['dist_map'][r.u, r.v] for r in self.net['paths_df'][self.net['paths_df']['path_id']==p].itertuples()) for p in rel_paths) / (amod_flow * 1000.0)
                pot_amod_time = sum(amod_res['x_ap'].get(p, 0) * (sum(congested_time_map[r.u, r.v] for r in self.net['paths_df'][self.net['paths_df']['path_id']==p].itertuples()) + AMOD_WAIT) for p in rel_paths) / amod_flow
            elif best_p:
                p_edges = self.net['paths_df'][self.net['paths_df']['path_id'] == best_p]
                pot_amod_dist = sum(self.net['dist_map'][r.u, r.v] for r in p_edges.itertuples()) / 1000.0
                pot_amod_time = sum(congested_time_map[r.u, r.v] for r in p_edges.itertuples()) + AMOD_WAIT
            else:
                pot_amod_dist, pot_amod_time = 0, 0

            path = self.net['pt_paths'][od][self.pt_path_choice[od]]
            pot_transit_wait = sum(line_wait_map[l] for l in path['gamma_odl'])
            walking_time = path['walking_dist'] / self.W_SPEED
            pot_transit_total_time = path['tau_pt'] + walking_time + pot_transit_wait

            # Price Recovery based on Logit Exact Reformulation
            v_pt = self.v_pt_fixed[od]
            amod_price = -(np.log(amod_flow) - np.log(d - amod_flow) + self.args.vot1 * pot_amod_time + v_pt)

            detailed_summary.append({
                'O': od[0], 'D': od[1],
                'Demand': d,
                'AMoD_Flow': round(amod_flow, 2),
                'Transit_Flow': round(transit_flow, 2),
                'Pot_AMoD_Time_min': round(pot_amod_time, 2),
                'Pot_AMoD_Dist_km': round(pot_amod_dist, 2),
                'AMoD_Price': round(amod_price, 2),
                'Pot_Transit_Wait_min': round(pot_transit_wait, 2),
                'Pot_Transit_Total_min': round(pot_transit_total_time, 2),
                'Transit_Path': self.pt_path_choice[od]
            })

        return pd.DataFrame(detailed_summary)
    
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
