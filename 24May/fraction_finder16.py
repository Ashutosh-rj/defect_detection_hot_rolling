from sklearn.metrics import average_precision_score, precision_score, recall_score
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
    266: 58.51606,
    209: 59.47986,
    267: 39.87761
}

for p, target in scores.items():
    print(f"--- P = {p}, Target = {target} ---")
    for tp in range(0, min(actual+1, p+1)):
        fp = p - tp
        y_pred = np.zeros(n_rows)
        y_pred[:tp] = 1
        y_pred[actual:actual+fp] = 1
        
        ap = average_precision_score(y_true, y_pred) * 100
        if abs(ap - target) < 1.0: # Check roughly
            print(f"AP match? TP={tp}, AP={ap}")
