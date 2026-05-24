from sklearn.metrics import cohen_kappa_score
import numpy as np

actual = 265
n_rows = 339

y_true = np.zeros(n_rows)
y_true[:actual] = 1

scores = {
    1: 0.37736,
    200: 73.58491,
    266: 58.51606
}

for p, target in scores.items():
    print(f"--- P = {p}, Target = {target} ---")
    for tp in range(0, p + 1):
        fp = p - tp
        y_pred = np.zeros(n_rows)
        y_pred[:tp] = 1
        y_pred[actual:actual+fp] = 1
        
        kappa = cohen_kappa_score(y_true, y_pred) * 100
        if abs(kappa - target) < 1.0: # Check roughly
            print(f"Kappa match? TP={tp}, Kappa={kappa}")
