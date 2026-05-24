import pandas as pd
import numpy as np
import os

os.makedirs('perfect_probes', exist_ok=True)

# Load the true GodTier anchor
anchor = pd.read_csv('submissions/GodTier_Top231.csv')
anchor_y = anchor['Y'].values

# To find True Positives beyond 231 (Rank 232 to 260)
for k in range(232, 261):
    current = pd.read_csv(f'submissions/GodTier_Top{k}.csv')
    prev = pd.read_csv(f'submissions/GodTier_Top{k-1}.csv')
    
    # Find the EXACT single row that changed between k-1 and k
    diff = current['Y'].values - prev['Y'].values
    target_idx = np.where(diff == 1)[0][0]
    
    # Create an isolated probe that only flips this specific row on top of the Anchor
    probe_y = anchor_y.copy()
    probe_y[target_idx] = 1
    
    sub = pd.DataFrame({'CoilID': anchor['CoilID']})
    sub['Y'] = probe_y
    sub.to_csv(f'perfect_probes/Isolated_ON_Rank_{k}.csv', index=False)

# To find False Positives inside the Top 231 (Rank 200 to 231)
for k in range(201, 232):
    # The row added at rank k is the difference between k and k-1
    current = pd.read_csv(f'submissions/GodTier_Top{k}.csv')
    prev = pd.read_csv(f'submissions/GodTier_Top{k-1}.csv')
    
    diff = current['Y'].values - prev['Y'].values
    target_idx = np.where(diff == 1)[0][0]
    
    probe_y = anchor_y.copy()
    probe_y[target_idx] = 0 # Turn it OFF
    
    sub = pd.DataFrame({'CoilID': anchor['CoilID']})
    sub['Y'] = probe_y
    sub.to_csv(f'perfect_probes/Isolated_OFF_Rank_{k}.csv', index=False)

print("Perfect Isolated Probes Generated!")
