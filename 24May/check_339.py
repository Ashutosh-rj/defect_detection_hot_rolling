import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

total = 339

def get_macro(P, N, tp, fp):
    fn = P - tp
    tn = N - fp
    y_true = np.array([1]*P + [0]*N)
    y_pred = np.array([1]*tp + [0]*fn + [1]*fp + [0]*tn)
    return f1_score(y_true, y_pred, average='macro') * 100

def get_micro(P, N, tp, fp):
    fn = P - tp
    tn = N - fp
    y_true = np.array([1]*P + [0]*N)
    y_pred = np.array([1]*tp + [0]*fn + [1]*fp + [0]*tn)
    return f1_score(y_true, y_pred, average='micro') * 100

def get_weighted(P, N, tp, fp):
    fn = P - tp
    tn = N - fp
    y_true = np.array([1]*P + [0]*N)
    y_pred = np.array([1]*tp + [0]*fn + [1]*fp + [0]*tn)
    return f1_score(y_true, y_pred, average='weighted') * 100

print("Searching exactly for 99.62264 when Total=339...")
found = False
for P in range(200, 330):
    N = total - P
    for tp in range(P-5, P+1):
        for fp in range(0, 6):
            if tp < 0 or fp > N: continue
            
            macro = get_macro(P, N, tp, fp)
            micro = get_micro(P, N, tp, fp)
            weighted = get_weighted(P, N, tp, fp)
            
            if abs(macro - 99.62264) < 0.0001:
                print(f"Match MACRO! P={P}, tp={tp}, fp={fp}, score={macro:.5f}")
                found = True
            if abs(micro - 99.62264) < 0.0001:
                print(f"Match MICRO! P={P}, tp={tp}, fp={fp}, score={micro:.5f}")
                found = True
            if abs(weighted - 99.62264) < 0.0001:
                print(f"Match WEIGHTED! P={P}, tp={tp}, fp={fp}, score={weighted:.5f}")
                found = True

if not found:
    print("NO COMBINATION FOUND FOR TOTAL=339.")
