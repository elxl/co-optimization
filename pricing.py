import numpy as np
import pandas as pd

def recover_amod_price(df_results, vot_user, transit_fare):
    """
    Recover AMoD OD price using Logit inversion consistent with Step 1.
    """
    prices = []
    for _, row in df_results.iterrows():
        x_a = max(1e-6, row['AMoD_Flow'])
        x_pt = max(1e-6, row['Transit_Flow'])
        t_a = row['Pot_AMoD_Time_min']
        t_pt = row["Pot_Transit_Total_min"]
        
        # Expanding utilities:
        # (-VOT * t_a - pi_od) - (-VOT * t_pt - PT) = ln(x_a / x_pt)
        if x_a == 0:
            pi_od = 1e9
        elif x_pt == 0:
            pi_od = -1e9
        else:
            logit_term = np.log(x_a / x_pt)
            time_diff_utility = vot_user * (t_pt - t_a)
            
            # Solve for pi_od
            pi_od = time_diff_utility + transit_fare - logit_term
        
        prices.append(round(pi_od, 2))
        
    df_results['AMoD_Price'] = prices
    return df_results           