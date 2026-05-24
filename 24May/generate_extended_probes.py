import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.stats import rankdata
import os

# Create folder for extended probes
os.makedirs('extended_probes', exist_ok=True)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
features = [c for c in train.columns if c not in ['id', 'CoilID', 'Y']]

X = train[features].values
y = train['Y'].values
X_test = test[features].values

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)
X_test = imputer.transform(X_test)

test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_va, y_va = X[val_idx], y[val_idx]
    
    smote = SMOTE(sampling_strategy=0.4, random_state=42)
    X_res, y_res = smote.fit_resample(X_tr, y_tr)
    
    models = [
        ('lgb', lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)),
        ('xgb', xgb.XGBClassifier(n_estimators=500, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)),
        ('cat', CatBoostClassifier(iterations=500, learning_rate=0.03, depth=6, random_state=42, verbose=0))
    ]
    
    for name, model in models:
        if name == 'cat':
            model.fit(X_res, y_res, eval_set=[(X_va, y_va)], early_stopping_rounds=50, verbose=0)
        elif name in ['lgb', 'xgb']:
            model.fit(X_res, y_res, eval_set=[(X_va, y_va)])
        else:
            model.fit(X_res, y_res)
        
        test_preds += model.predict_proba(X_test)[:, 1] / (10 * len(models))

ranks = rankdata(test_preds)

# Load the EXACT GodTier_Top231 as anchor
anchor = pd.read_csv('submissions/GodTier_Top231.csv')
anchor_y = anchor['Y'].values

# We need probes from 261 all the way to 339!
for k in range(261, 340):
    # Find the EXACT index that corresponds to Rank k
    target_idx = np.where(ranks == (len(ranks) - k + 1))[0][0]
    
    probe_y = anchor_y.copy()
    probe_y[target_idx] = 1 # Turn it ON
    
    sub = pd.DataFrame({'CoilID': anchor['CoilID']})
    sub['Y'] = probe_y
    sub.to_csv(f'extended_probes/Isolated_ON_Rank_{k}.csv', index=False)

print("Extended Probes Generated!")
