"""
No-subsidy baseline: run AMoD best response against the real-world GTFS-derived
transit frequencies (instead of the social-optimum frequencies) and evaluate the
resulting social cost. Same arg interface as main.py; skips Social Optimum and
Transit Best Response.

Single-cell usage:
    python baseline_amod_br.py --bl 1.0 --ca 1.0 --cl 320 --theta 0.5

A grid wrapper (shell script for cluster) will be added separately.
"""
import argparse
import os
import pickle
import time

from src.data_loader import load_nyc_network
from src.amod_br import AMoD_BestResponse
from src.social_optimum import evaluate_baseline_social_cost
from src.constants import TRANSIT_FARE


def main():
    parser = argparse.ArgumentParser(description="No-subsidy baseline: AMoD best response with GTFS frequencies")
    # Mirrors main.py arg block
    parser.add_argument('-o', "--output_dir", type=str, default='./results/baseline_amod_br', help='Directory for result files')
    parser.add_argument("--input_dir", type=str, default='./data_nyc/processed', help='Directory for input files')
    parser.add_argument("--demand_file", type=str, default='demand.csv', help='Demand file (relative to input_dir)')
    parser.add_argument("--transit_file", type=str, default='metro/optimization_inputs.pkl.gz', help='Transit pickle (relative to input_dir)')
    parser.add_argument("--amod_file", type=str, default='on_demand/paths.csv', help='AMoD paths file (relative to input_dir)')
    parser.add_argument("--amod_map", type=str, default='on_demand/map.csv', help='AMoD map file (relative to input_dir)')
    parser.add_argument("--freq_file", type=str, default='./data_nyc/processed/metro/gtfs_frequencies.pkl',
                        help='Pickle of {line_id: freq_vph} produced by extract_gtfs_frequencies cell in data_explore_nyc.ipynb')
    parser.add_argument("--bl", type=float, default=1.0, help='Background traffic level')
    parser.add_argument("--ca", type=float, default=1.0, help='AMoD operation cost per km')
    parser.add_argument("--cl", type=float, default=320, help='Metro operation cost per line per hour')
    parser.add_argument("--pt_cap", type=int, default=900, help='Transit capacity (passengers per departure)')
    parser.add_argument("--walk", type=float, default=3, help='Walking speed (mph)')
    parser.add_argument("--num_path", type=int, default=1, help='Number of candidate transit paths per OD (1 = original single-path model)')
    parser.add_argument("--vot", type=float, default=0.4, help='Value of time for road users (drivers)')
    parser.add_argument("--vot1", type=float, default=0.3, help='Value of time for passengers')
    parser.add_argument("--mip", type=float, default=0.01, help='Optimization gap.')
    parser.add_argument("--alpha", type=float, default=0.15, help='alpha parameter of BPR.')
    parser.add_argument("--beta", type=float, default=4, help='beta parameter of BPR.')
    parser.add_argument("--omega", type=float, default=0.0, help='Coefficient for AMoD social benefit per passenger.')
    parser.add_argument("--reg", type=int, default=2, help='Regularization. (0: no, 1: quadratic, 2: entropy)')
    parser.add_argument("--theta", type=float, default=0.5, help='Coefficient for regularization term')
    parser.add_argument("--plot", type=bool, default=False, help='Whether to plot SCA convergence online.')
    args = parser.parse_args()

    reg_dict = {0: "", 1: "_quad", 2: "_ent"}
    if args.reg != 0:
        suffix = f"_{args.bl}_{args.ca}_{args.cl}{reg_dict[args.reg]}_theta{args.theta}"
    else:
        suffix = f"_{args.bl}_{args.ca}_{args.cl}"

    os.makedirs(args.output_dir, exist_ok=True)

    t_start = time.time()
    print("--- 1. Loading Network Data ---")
    net_data, demand = load_nyc_network(args)
    print(f"Loaded {len(net_data['OD_pairs'])} OD pairs and {len(net_data['E_a'])} edges.")

    # ----------------------
    # Baseline transit strategy: GTFS frequencies (snapped to S_SET)
    # ----------------------
    with open(args.freq_file, "rb") as f:
        gtfs_freq = pickle.load(f)
    # Sanity-check: every line in net_data['L'] must have a frequency entry.
    missing = [l for l in net_data['L'] if l not in gtfs_freq]
    if missing:
        raise RuntimeError(f"freq_file is missing {len(missing)} lines from net_data['L']: {missing[:5]}...")
    baseline_transit = {'frequencies': gtfs_freq, 'fare': TRANSIT_FARE}

    n_served = sum(1 for l in net_data['L'] if gtfs_freq[l] > 0)
    print(f"Loaded GTFS frequencies for {len(gtfs_freq)} lines ({n_served} with service > 0).")

    # ----------------------
    # AMoD best response against fixed GTFS frequencies
    # ----------------------
    print("\n--- 2. Solving AMoD Best Response (GTFS-fixed transit) ---")
    amod_br = AMoD_BestResponse(args, net_data, demand, baseline_transit)

    profit_ub = amod_br.solve_upper_bound_relaxation()
    if profit_ub is None:
        raise RuntimeError("AMoD-BR relaxation failed for the baseline run.")
    print(f"AMoD Best Response Upper Bound (Relaxation): {profit_ub:.2f}")

    t_abr_start = time.time()
    print("\nRunning SCA to find local optimum (lower bound)...")
    # Cold start: no SO solution to warm-start from in the baseline scenario.
    profit_lb, f_opt, r_opt, x_a_opt, x_ap_opt, sca_converge = amod_br.solve_sca()
    t_abr_end = time.time()
    sca_converge['total_time'] = t_abr_end - t_abr_start
    sca_converge['upper_bound'] = profit_ub
    if profit_lb is None:
        raise RuntimeError("AMoD-BR SCA failed for the baseline run.")
    if profit_ub != float('inf'):
        gap_abs = profit_ub - profit_lb
        gap_pct = gap_abs / abs(profit_lb) * 100 if profit_lb != 0 else 0
        sca_converge['optimality_gap'] = gap_pct
        print(f"AMoD Profit (LB): {profit_lb:.2f}  |  Gap: ${gap_abs:.2f} ({gap_pct:.2f}%)")
    else:
        print(f"AMoD Profit (LB): {profit_lb:.2f}")
    print(f"SCA completed in {(t_abr_end - t_abr_start)/60:.2f} min.")

    # Save SCA convergence (same filename convention as the with-subsidy AMoD-BR runs)
    with open(os.path.join(args.output_dir, f"sca_convergence{suffix}.pkl"), "wb") as fp:
        pickle.dump(sca_converge, fp)

    amod_state = {'f': f_opt, 'r': r_opt, 'x_a': x_a_opt, 'x_ap': x_ap_opt}

    # OD-level mode-split for the baseline run
    df_amod_br_results = amod_br.get_results_dataframe(amod_state, baseline_transit)
    df_amod_br_results.to_csv(os.path.join(args.output_dir, f"baseline_mode_split{suffix}.csv"), index=False)

    # ----------------------
    # Social cost at the baseline
    # ----------------------
    print("\n--- 3. Evaluating Social Cost at Baseline ---")
    df_summary = evaluate_baseline_social_cost(args, net_data, demand, amod_state, gtfs_freq)
    df_summary["AMoD_Profit"] = round(profit_lb, 2)
    df_summary["AMoD_Profit_UB"] = round(profit_ub, 2) if profit_ub != float('inf') else None
    df_summary["Total_Time_min"] = round((time.time() - t_start) / 60.0, 2)
    out_summary = os.path.join(args.output_dir, f"baseline_summary{suffix}.csv")
    df_summary.to_csv(out_summary, index=False)

    print(df_summary.T.to_string())
    print(f"\nTotal runtime: {(time.time() - t_start)/60:.2f} min")
    print(f"Saved -> {out_summary}")


if __name__ == "__main__":
    main()
