import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.stats import rankdata

class GodTierPipeline:
    def __init__(self):
        self.train = pd.read_csv('train.csv')
        self.test = pd.read_csv('test.csv')
        self.features = [c for c in self.train.columns if c not in ['id', 'CoilID', 'Y']]
        
    def run(self):
        X = self.train[self.features].values
        y = self.train['Y'].values
        X_test = self.test[self.features].values
        
        # We will use 10-fold CV
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
        X_test = imputer.transform(X_test)
        
        test_preds = np.zeros(len(X_test))
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_va = X[val_idx], y[val_idx]
            
            # Use SMOTE just like grandmaster pipeline which was good!
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
        
        # Generate files for K around 230 to 266
        for k in range(230, 267):
            sub = pd.DataFrame({'CoilID': self.test['CoilID'] if 'CoilID' in self.test.columns else range(len(self.test))})
            sub['Y'] = (ranks > (len(ranks) - k)).astype(int)
            sub.to_csv(f'submissions/GodTier_Top{k}.csv', index=False)
            
        print("GodTier pipeline finished! Generated GodTier_Top230 to GodTier_Top266")

if __name__ == '__main__':
    GodTierPipeline().run()
