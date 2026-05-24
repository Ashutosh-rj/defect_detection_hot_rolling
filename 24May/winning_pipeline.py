import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X_train = train.drop(columns=['Y', 'CoilID'] if 'CoilID' in train.columns else ['Y'])
y_train = train['Y']
X_test = test.drop(columns=['CoilID'] if 'CoilID' in test.columns else [])

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

# Ensemble of LightGBM, XGBoost, CatBoost
models = [
    lgb.LGBMClassifier(random_state=42, n_estimators=500, learning_rate=0.05),
    xgb.XGBClassifier(random_state=42, n_estimators=500, learning_rate=0.05, eval_metric='logloss'),
    CatBoostClassifier(random_state=42, iterations=500, learning_rate=0.05, verbose=False)
]

# 5-Fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test_imp))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_imp, y_train)):
    X_tr, y_tr = X_train_imp[train_idx], y_train.iloc[train_idx]
    
    fold_preds = np.zeros(len(X_test_imp))
    for model in models:
        model.fit(X_tr, y_tr)
        fold_preds += model.predict_proba(X_test_imp)[:, 1] / len(models)
        
    test_preds += fold_preds / 5

# Threshold = 0.05
y_test_pred = (test_preds > 0.05).astype(int)

print(f"Number of predicted positives (threshold 0.05): {sum(y_test_pred)}")

sub = pd.DataFrame({'CoilID': test['CoilID'] if 'CoilID' in test.columns else range(len(test)), 'Y': y_test_pred})
sub.to_csv('expected_submission.csv', index=False)
print("Saved to expected_submission.csv")
