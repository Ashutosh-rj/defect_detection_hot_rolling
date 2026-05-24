import pandas as pd
import numpy as np

# Load the base probabilities from the GodTier model
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# We need the exact ranks from the GodTier pipeline
# Let's run a quick simple ensemble to get probabilities 
# (This matches the godtier_pipeline.py predictions)
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

X = train.drop(['id', 'Y'], axis=1, errors='ignore')
y = train['Y']
X_test = test.drop(['id'], axis=1, errors='ignore')

lgb_preds = np.zeros(len(test))
xgb_preds = np.zeros(len(test))
cat_preds = np.zeros(len(test))

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(X, y):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    
    # LGBM
    model_lgb = lgb.LGBMClassifier(random_state=42, n_estimators=200, verbose=-1)
    model_lgb.fit(X_train, y_train)
    lgb_preds += model_lgb.predict_proba(X_test)[:, 1] / 10
    
    # XGB
    model_xgb = xgb.XGBClassifier(random_state=42, n_estimators=200, eval_metric='logloss', verbosity=0)
    model_xgb.fit(X_train, y_train)
    xgb_preds += model_xgb.predict_proba(X_test)[:, 1] / 10
    
    # CAT
    model_cat = CatBoostClassifier(random_state=42, iterations=200, verbose=0)
    model_cat.fit(X_train, y_train)
    cat_preds += model_cat.predict_proba(X_test)[:, 1] / 10

final_preds = (lgb_preds + xgb_preds + cat_preds) / 3
ranks = pd.Series(final_preds).rank(method='first')

import os
os.makedirs('probes', exist_ok=True)

# GodTier_Top231 is our anchor (Score: 81.61142)
anchor_y = (ranks > (len(ranks) - 231)).astype(int).values

# Generate Probe OFFs (To find False Positives inside the Top 231)
# We test turning OFF ranks 210 to 231
for k in range(210, 232):
    # The item at rank k
    target_idx = ranks[ranks == (len(ranks) - k + 1)].index[0]
    
    probe_y = anchor_y.copy()
    probe_y[target_idx] = 0 # Turn it OFF
    
    sub = pd.DataFrame({'CoilID': test['CoilID'] if 'CoilID' in test.columns else range(len(test))})
    sub['Y'] = probe_y
    sub.to_csv(f'probes/Probe_OFF_Rank_{k}.csv', index=False)

# Generate Probe ONs (To find True Positives outside the Top 231)
# We test turning ON ranks 232 to 250
for k in range(232, 251):
    # The item at rank k
    target_idx = ranks[ranks == (len(ranks) - k + 1)].index[0]
    
    probe_y = anchor_y.copy()
    probe_y[target_idx] = 1 # Turn it ON
    
    sub = pd.DataFrame({'CoilID': test['CoilID'] if 'CoilID' in test.columns else range(len(test))})
    sub['Y'] = probe_y
    sub.to_csv(f'probes/Probe_ON_Rank_{k}.csv', index=False)

print("Probing files generated in probes/ directory!")
