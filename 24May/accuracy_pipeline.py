import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

features = [c for c in train.columns if c not in ['Y', 'CoilID']]

X = train[features].fillna(0)
y = train['Y']
X_test = test[features].fillna(0)

print(f"Baseline (all zeros) accuracy on train: {(y == 0).mean():.4f}")

seeds = [42, 43, 44]
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

for seed in seeds:
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=seed)
    
    models_dict = {
        'lgb': lgb.LGBMClassifier(verbose=-1, n_estimators=200, random_state=seed),
        'xgb': xgb.XGBClassifier(verbosity=0, n_estimators=200, random_state=seed),
        'cat': CatBoostClassifier(verbose=0, iterations=200, random_state=seed),
        'et': ExtraTreesClassifier(n_estimators=200, random_state=seed),
        'rf': RandomForestClassifier(n_estimators=200, random_state=seed)
    }
    
    for name, model in models_dict.items():
        for train_idx, val_idx in cv.split(X, y):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_va = X.iloc[val_idx]
            
            model.fit(X_tr, y_tr)
            oof_preds[val_idx] += model.predict_proba(X_va)[:, 1] / (len(seeds) * len(models_dict))
            test_preds += model.predict_proba(X_test)[:, 1] / (5 * len(seeds) * len(models_dict))

# Find best threshold for Accuracy on OOF
best_thresh = 0.5
best_acc = 0
for thresh in np.arange(0.1, 0.9, 0.01):
    acc = accuracy_score(y, (oof_preds > thresh).astype(int))
    if acc > best_acc:
        best_acc = acc
        best_thresh = thresh

print(f"Best OOF Accuracy: {best_acc:.4f} at threshold: {best_thresh:.2f}")

# Create final submission using best accuracy threshold
sub_acc = pd.DataFrame({'CoilID': test['CoilID'] if 'CoilID' in test.columns else range(len(test))})
sub_acc['Y'] = (test_preds > best_thresh).astype(int)
sub_acc.to_csv('submissions/Best_Accuracy_Submission.csv', index=False)
print(f"Saved Best_Accuracy_Submission.csv with {sub_acc['Y'].sum()} positives.")

# Also let's try the default 0.5 threshold just in case
sub_default = pd.DataFrame({'CoilID': test['CoilID'] if 'CoilID' in test.columns else range(len(test))})
sub_default['Y'] = (test_preds > 0.5).astype(int)
sub_default.to_csv('submissions/Default_0.5_Submission.csv', index=False)
print(f"Saved Default_0.5_Submission.csv with {sub_default['Y'].sum()} positives.")
