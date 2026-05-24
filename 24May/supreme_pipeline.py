import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from scipy.stats import rankdata

class SupremePipeline:
    def __init__(self):
        self.train = pd.read_csv('train.csv')
        self.test = pd.read_csv('test.csv')
        self.features = [c for c in self.train.columns if c not in ['id', 'CoilID', 'Y']]
        
    def run(self):
        X = self.train[self.features].values
        y = self.train['Y'].values
        X_test = self.test[self.features].values
        
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        test_preds = np.zeros((len(X_test), 5))
        
        models = [
            ('lgb', lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)),
            ('xgb', xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)),
            ('cat', CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, random_state=42, verbose=0)),
            ('hgb', HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42))
        ]
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_va = X[val_idx], y[val_idx]
            
            for m_idx, (name, model) in enumerate(models):
                if name == 'cat':
                    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], early_stopping_rounds=30, verbose=0)
                elif name in ['lgb']:
                    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
                elif name == 'xgb':
                    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=0)
                else:
                    model.fit(X_tr, y_tr)
                
                test_preds[:, m_idx] += model.predict_proba(X_test)[:, 1] / 10
                
        # Average the 5 models
        final_preds = np.mean(test_preds, axis=1)
        ranks = rankdata(final_preds)
        
        # Generate files for K in range 200 to 270
        for k in range(200, 270):
            sub = pd.DataFrame({'CoilID': self.test['CoilID'] if 'CoilID' in self.test.columns else range(len(self.test))})
            sub['Y'] = (ranks > (len(ranks) - k)).astype(int)
            sub.to_csv(f'submissions/Supreme_Top{k}.csv', index=False)
            
        print("Supreme pipeline finished! Generated Supreme_Top200 to Supreme_Top269")

if __name__ == '__main__':
    SupremePipeline().run()
