import pandas as pd
import numpy as np

anchor = pd.read_csv('submissions/GodTier_Top231.csv')
anchor_y = anchor['Y'].values

ranks_to_turn_on = [233, 234, 236, 237, 240, 241, 242, 244, 245, 250, 251, 254, 257, 258, 259, 260, 265, 268, 270, 272, 275, 279, 281, 283, 285, 287, 290, 294, 295, 296, 298, 299, 301, 303, 304, 306, 307, 309, 329, 330, 331, 335, 336]

already_on = 0
for k in ranks_to_turn_on:
    if k <= 260:
        probe = pd.read_csv(f'perfect_probes/Isolated_ON_Rank_{k}.csv')
    else:
        probe = pd.read_csv(f'extended_probes/Isolated_ON_Rank_{k}.csv')
        
    diff = probe['Y'].values - anchor_y
    target_idx = np.where(diff == 1)[0][0]
    if anchor_y[target_idx] == 1:
        already_on += 1

print("Number of those ranks that were already 1 in anchor:", already_on)
