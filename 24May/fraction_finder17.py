from sklearn.metrics import matthews_corrcoef
import numpy as np
import math

n_rows = 339
actual = 265
target = 60.29577

print("Testing MCC for Supreme_Top202 (P=202)")
p = 202
for tp in range(0, p + 1):
    fp = p - tp
    fn = actual - tp
    tn = n_rows - tp - fp - fn
    
    y_true = np.zeros(n_rows)
    y_true[:actual] = 1
    
    y_pred = np.zeros(n_rows)
    y_pred[:tp] = 1
    y_pred[actual:actual+fp] = 1
    
    mcc = matthews_corrcoef(y_true, y_pred) * 100
    if abs(mcc - target) < 1.0:
        print(f"MCC Match? TP={tp}, MCC={mcc}")
        
    # Also test Cohen's Kappa
    from sklearn.metrics import cohen_kappa_score
    kappa = cohen_kappa_score(y_true, y_pred) * 100
    if abs(kappa - target) < 1.0:
        print(f"Kappa Match? TP={tp}, Kappa={kappa}")
