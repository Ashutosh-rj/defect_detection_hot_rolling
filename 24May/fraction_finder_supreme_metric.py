from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, matthews_corrcoef, balanced_accuracy_score, cohen_kappa_score
import numpy as np
import warnings
warnings.filterwarnings('ignore')

n_rows = 339
p = 202
target = 60.29577

print(f"Finding metric for P={p}, Target={target}")

for actual in range(1, 339):
    y_true = np.zeros(n_rows)
    y_true[:actual] = 1
    
    for tp in range(0, min(actual+1, p+1)):
        fp = p - tp
        fn = actual - tp
        tn = n_rows - tp - fp - fn
        
        y_pred = np.zeros(n_rows)
        y_pred[:tp] = 1
        y_pred[actual:actual+fp] = 1
        
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred) * 100,
            'Precision': precision_score(y_true, y_pred) * 100 if p > 0 else 0,
            'Recall': recall_score(y_true, y_pred) * 100 if actual > 0 else 0,
            'F1': f1_score(y_true, y_pred) * 100,
            'Macro_F1': f1_score(y_true, y_pred, average='macro') * 100,
            'Weighted_F1': f1_score(y_true, y_pred, average='weighted') * 100,
            'Jaccard': jaccard_score(y_true, y_pred) * 100,
            'Macro_Jaccard': jaccard_score(y_true, y_pred, average='macro') * 100,
            'MCC': matthews_corrcoef(y_true, y_pred) * 100,
            'Balanced_Acc': balanced_accuracy_score(y_true, y_pred) * 100,
            'Kappa': cohen_kappa_score(y_true, y_pred) * 100
        }
        
        for name, val in metrics.items():
            if abs(val - target) < 1e-4:
                print(f"BINGO! Metric={name}, Actual={actual}, TP={tp}, Val={val}")
