import numpy as np
from sklearn.metrics import f1_score

total = 339

def get_macro(P, N, tp, fp):
    fn = P - tp
    tn = N - fp
    y_true = np.array([1]*P + [0]*N)
    y_pred = np.array([1]*tp + [0]*fn + [1]*fp + [0]*tn)
    return f1_score(y_true, y_pred, average='macro') * 100

for P in range(200, 300):
    N = total - P
    for tp in range(150, P+1):
        fp = 231 - tp
        if fp < 0 or fp > N: continue
        macro = get_macro(P, N, tp, fp)
        if abs(macro - 81.61142) < 0.0001:
            print(f"Match MACRO! P={P}, tp={tp}, fp={fp}, score={macro:.5f}")
