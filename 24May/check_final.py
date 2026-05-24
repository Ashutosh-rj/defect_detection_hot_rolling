import pandas as pd

anchor = pd.read_csv('submissions/GodTier_Top231.csv')
final = pd.read_csv('FINAL_100_SCORE.csv')

print(f"Anchor 1s: {anchor['Y'].sum()}")
print(f"Final 1s: {final['Y'].sum()}")

diff = final['Y'].values - anchor['Y'].values
print(f"Added 1s: {(diff == 1).sum()}")
print(f"Removed 1s: {(diff == -1).sum()}")

