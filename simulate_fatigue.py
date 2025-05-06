import numpy as np
import pandas as pd

def stress_to_life(sigma_a, sn_curve_df):
    # Interpolates S-N curve to get Nf
    return np.interp(sigma_a, sn_curve_df["stress"], sn_curve_df["life"])

def simulate_damage(F, L, I, E, sn_curve_df, cycles_per_step=100, total_steps=1000):
    M = F * L
    sigma_max = M * L / I  # simplified max stress (cantilever tip)
    Nf = stress_to_life(sigma_max, sn_curve_df)
    
    D = 0
    damage_history = []
    for i in range(total_steps):
        D += cycles_per_step / Nf
        damage_history.append(min(D, 1.0))
        if D >= 1.0:
            break
    return np.array(damage_history)
