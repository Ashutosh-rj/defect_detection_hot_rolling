from sklearn.metrics import jaccard_score
import numpy as np

actual = 265
n_rows = 339

y_true = np.zeros(n_rows)
y_true[:actual] = 1

scores = {
    1: 0.37736,
    3: 1.13208,
    200: 73.58491,
    202: 76.2264,
    266: 58.51606
}

for p, target in scores.items():
    print(f"--- P = {p}, Target = {target} ---")
    for tp in range(0, min(actual+1, p+1)):
        fp = p - tp
        y_pred = np.zeros(n_rows)
        y_pred[:tp] = 1
        y_pred[actual:actual+fp] = 1
        
        j_macro = jaccard_score(y_true, y_pred, average='macro') * 100
        if abs(j_macro - target) < 1.0: # Check roughly
            print(f"Jaccard Macro match? TP={tp}, J_macro={j_macro}")
