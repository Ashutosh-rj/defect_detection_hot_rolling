import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split

anchor = pd.read_csv('submissions/GodTier_Top231.csv')
anchor_y = anchor['Y'].values

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
features = [c for c in train.columns if c not in ['id', 'CoilID', 'Y']]

X = train[features].values
y = train['Y'].values
X_test = test[features].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)
X_test = imputer.transform(X_test)

from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy=0.4, random_state=42)
X_res, y_res = smote.fit_resample(X, y)

model1 = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, max_depth=6, random_state=42, verbose=-1)
model2 = xgb.XGBClassifier(n_estimators=500, learning_rate=0.03, max_depth=6, random_state=42, verbosity=0)

model1.fit(X_res, y_res)
model2.fit(X_res, y_res)

test_preds = (model1.predict_proba(X_test)[:, 1] + model2.predict_proba(X_test)[:, 1]) / 2

ones_indices = np.where(anchor_y == 1)[0]
probs_of_ones = test_preds[ones_indices]

sorted_args = np.argsort(probs_of_ones)
least_confident_indices = ones_indices[sorted_args]

print("\n--- CHECK THESE FILES IN deep_off_probes ---")
for idx in least_confident_indices[:20]:
    print(f"Isolated_OFF_Row_{idx}.csv")
print("--------------------------------------------\n")
