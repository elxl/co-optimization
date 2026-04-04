import argparse
import os
import pickle
import time
import pandas as pd
from src.data_loader import load_nyc_network
from src.social_optimum import SocialOptimum
from src.amod_br import AMoD_BestResponse
from src.transit_br import TransitBestResponse

def main():
    parser = argparse.ArgumentParser(description="Multimodal Game Theory Pipeline")
    parser.add_argument('-o', "--output_dir", type=str, default='./results', help='Directory for result files')
    parser.add_argument("--input_dir", type=str, default='./data_nyc/processed', help='Directory for input files')
    parser.add_argument("--demand_file", type=str, default='demand.csv', help='Directory for demand file')
    parser.add_argument("--transit_file", type=str, default='metro/optimization_inputs.pkl.gz', help='Directory for transit file')
    parser.add_argument("--amod_file", type=str, default='on_demand/paths.csv', help='Directory for amod file')
    parser.add_argument("--amod_map", type=str, default='on_demand/map.csv', help='Directory for amod map file')
    parser.add_argument("--bl", type=float, default=1.0, help='Background traffic level')
    parser.add_argument("--ca", type=float, default=1.0, help='AMoD operation cost per km')
    parser.add_argument("--cl", type=float, default=100, help='Metro operation cost per line per hour')
    parser.add_argument("--walk", type=float, default=3.6, help='Walking speed (mph)')
    parser.add_argument("--vot", type=float, default=0.4, help='Environmental price for driving')
    parser.add_argument("--vot1", type=float, default=0.3, help='Value of time for passengers')
    parser.add_argument("--mip", type=float, default=0.01, help='Optimization gap.')
    parser.add_argument("--alpha", type=float, default=0.15, help='alpha parameter of BPR.')
    parser.add_argument("--beta", type=float, default=4, help='beta parameter of BRP.')
    parser.add_argument("--omega", type=float, default=0.0, help='Coefficient for AMoD social benefit per passenger.')
    parser.add_argument("--reg", type=int, default=2, help='Regularization. (0: no regularization, 1: quadratic, 2: entropy)')
    parser.add_argument("--theta", type=float, default=0.2, help='Coefficient for regularization term (higher theta = more spread out; lower theta = closer to original objective)')
    parser.add_argument("--load_so", type=str, default=None, help='File to load the social optimum solution from instead of solving the optimization problems.')
    parser.add_argument("--plot", type=bool, default=False, help='Whether to generate convergence plot online for the AMoD Best Response SCA algorithm.')
    args = parser.parse_args()

    reg_dict = {0:"", 1:"_quad", 2:"_ent"}
    if args.reg != 0:
        suffix = f"_{args.bl}_{args.ca}_{args.cl}{reg_dict[args.reg]}_theta{args.theta}"
    else:
        suffix = f"_{args.bl}_{args.ca}_{args.cl}"

    t_start = time.time()
    print("--- 1. Loading Network Data ---")
    net_data, demand = load_nyc_network(args)
    print(f"Loaded {len(net_data['OD_pairs'])} OD pairs and {len(net_data['E_a'])} edges.")

    # ----------------------
    # Social Optimum
    # ----------------------
    if args.load_so:
        print(f"\n---2. Loading Social Optimum Solution from {args.load_so} ---")
        with open(args.load_so, "rb") as f:
            so_save_data = pickle.load(f)
        so_cost = so_save_data['so_cost']
        target_transit_strategy = so_save_data['target_transit_strategy']
        target_amod_strategy = so_save_data['target_amod_strategy']
        amod_so_state = so_save_data['amod_so_state']
        transit_so_state = so_save_data['transit_so_state']
        
        print(f"Loaded successfully! Minimum Social Cost was: {so_cost:.2f}")
    else:
        print("\n--- 2. Solving Target Strategy (Social Optimum) ---")
        so_model = SocialOptimum(args, net_data, demand)
        so_obj, target_transit_strategy, target_amod_strategy = so_model.solve()
        t_so = time.time()
        
        if so_obj is not None:
            print(f"Social Optimum solved!")
            # Generate summary of so solution and save to file
            print("Recovering prices and generating mode split summary...")
            df_so_results = so_model.get_results_dataframe()
            # df_so_summary = so_model.get_summary()
            # df_so_summary.to_csv(os.path.join(args.output_dir, f"so_summary{suffix}.csv"), index=False)
            df_so_results.to_csv(os.path.join(args.output_dir, f"mode_split{suffix}.csv"), index=False)

            # ---------------------------------------------------------
            # Extract the operators' strategies at the Social Optimum
            # ---------------------------------------------------------
            price_dict = df_so_results.set_index(['O', 'D'])['AMoD_Price'].to_dict()
            amod_so_state = {
                'r': target_amod_strategy['r'],
                'f': target_amod_strategy['f'],
                'x_a': {od: so_model.x_a[od].X for od in net_data['OD_pairs']},
                'x_ap': {p: so_model.x_ap[p].X for p in so_model.x_ap.keys()},
                'pi_a': price_dict
            }
            
            transit_so_state = {
                'frequencies': target_transit_strategy['frequencies'],
                'fare': target_transit_strategy['fare'],
                'x_pt': {od: so_model.x_pt[od].X for od in net_data['OD_pairs']}
            }

            # Save so solution to file for future loading (debugging)
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
                
            so_save_path = os.path.join(args.output_dir, f"so_solution{suffix}.pkl")
            so_save_data = {
                'so_cost': so_obj,
                'target_transit_strategy': target_transit_strategy,
                'target_amod_strategy': target_amod_strategy,
                'amod_so_state': amod_so_state,
                'transit_so_state': transit_so_state
            }
            with open(so_save_path, 'wb') as f:
                pickle.dump(so_save_data, f)
            print(f"Saved Social Optimum states to {so_save_path}")
        else:
            raise RuntimeError("Social Optimum optimization failed to find a solution. Please check the model and data. If you have a previously solved solution, you can load it using the --load_so argument.")

    # ----------------------
    # AMoD best response
    # ----------------------
    print("\n--- 3. Solving AMoD Best Response ---")
    amod_br = AMoD_BestResponse(args, net_data, demand, target_transit_strategy)

    # Solve convex relaxation for upper bound
    profit_ub = amod_br.solve_upper_bound_relaxation()
    if profit_ub is not None:
        print(f"AMoD Best Response Upper Bound (Relaxation): {profit_ub:.2f}")
    else:
        raise RuntimeError("AMoD Best Response relaxation failed to find a solution. Please check the model and data.")
    
    # Pass the Social Optimum flows as the starting point for the SCA algorithm
    print("\nRunning SCA to find local optimum (lower bound)...")
    profit_lb, f_opt, r_opt = amod_br.solve_sca(
        f_initial=target_amod_strategy['f'], 
        r_initial=target_amod_strategy['r'],
        x_a_initial=amod_so_state['x_a'],
        x_ap_initial=amod_so_state['x_ap']
    )
    t_abr = time.time()
    
    if profit_lb is not None:
        print(f"\nAMoD Maximum Profit (Lower Bound): {profit_lb:.2f}")
        if profit_ub != float('inf'):
            gap_abs = profit_ub - profit_lb
            gap_pct = gap_abs / abs(profit_lb) * 100 if profit_lb != 0 else 0
            print(f"[Result] Optimality Gap: ${gap_abs:.2f} ({gap_pct:.2f}%)")
    else:
        raise RuntimeError("AMoD Best Response optimization failed to find a solution. Please check the model and data.")

    # ----------------------
    # Transit best response
    # ----------------------
    # Get AMoD prices
    if args.load_so and os.path.exists(args.load_so):
        price_dict = so_save_data['amod_so_state']['pi_a']
    else:
        price_dict = df_so_results.set_index(['O', 'D'])['AMoD_Price'].to_dict()
    
    amod_fixed_state = {
        'f': target_amod_strategy['f'],
        'x_a': amod_so_state['x_a'],
        'x_ap': amod_so_state['x_ap'],
        'pi_a': price_dict
    }
    # print("\n--- 4. Solving Transit Best Response (MICP) ---")
    transit_br = TransitBestResponse(args, net_data, demand, amod_fixed_state, transit_so_state)
    transit_br_obj, transit_opt_strategy = transit_br.solve()
    t_ptbr = time.time()
    
    if transit_br_obj is not None:
        print(f"\n[Result] Transit Maximum Deviated Objective: {transit_br_obj:.2f}")
        
        # ---------------------------------------------------------
        # Evaluate Target State & Calculate k-Implementation Payments
        # ---------------------------------------------------------
        print("\n--- 5. Evaluating k-Implementation Subsidies ---")

        # Evaluate the utilities of both operators at the Social Optimum state
        amod_so_profit = amod_br.evaluate(amod_fixed_state)
        transit_so_obj = transit_br.evaluate(transit_so_state)

        print(f"AMoD Profit at Target Strategy: {amod_so_profit:.2f}")
        print(f"Transit Objective at Target Strategy: {transit_so_obj:.2f}")

        # Calculate the final subsidies (V = Deviation Profit - Target Profit)
        # We use max(0, gap) because if the target profit is already higher than 
        # the deviation profit, the operator doesn't need a subsidy to stay!
        v_a = max(0, profit_lb - amod_so_profit)
        v_pt = max(0, transit_br_obj - transit_so_obj)
        
        print("\n==================================================")
        print(f"=> Total Subsidy: ${v_a + v_pt:.2f}")
        print(f"=> City Subsidy to AMoD (V_a):     ${v_a:.2f}")
        print(f"=> City Subsidy to Transit (V_pt): ${v_pt:.2f}")
        print("==================================================")

        t_end = time.time()
        print(f"\nTotal runtime: {(t_end - t_start)/60:.2f} minutes")

        k_summary = {"bl": args.bl, "ca": args.ca, "cl": args.cl,
                    "AMoD_so": amod_so_profit, "Transit_so": transit_so_obj, "AMoD_br": profit_lb, "AMoD_br_gap": gap_pct,
                    "Transit_br": transit_br_obj, "V_a": v_a, "V_pt": v_pt, 'Total_V': v_a + v_pt, 'Social_cost':so_save_data['so_cost'], 
                    't_so': t_so - t_start, 't_amod_br': t_abr - t_so, 't_transit_br': t_ptbr - t_abr, 't_total': t_end - t_start
                    }
        df_k_summary = pd.DataFrame([k_summary])
        df_k_summary.to_csv(os.path.join(args.output_dir, f"k_summary{suffix}.csv"), index=False)
    
    else:
        raise RuntimeError("Transit Best Response optimization failed to find a solution. Please check the model and data.")

if __name__ == "__main__":
    main()