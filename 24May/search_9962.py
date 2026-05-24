import numpy as np
from sklearn.metrics import f1_score

def get_f1(P, N, tp, fp):
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

P = 265
N = 339 - 265

print(get_f1(265, 74, 264, 0))
print(get_f1(265, 74, 265, 1))
print(get_micro(265, 74, 264, 0))
print(get_micro(265, 74, 265, 1))

print("Searching...")
for P_test in range(250, 300):
    for N_test in range(20, 150):
        for tp in range(P_test-5, P_test+1):
            for fp in range(0, 5):
                score = get_micro(P_test, N_test, tp, fp)
                if abs(score - 99.62264) < 0.001:
                    print(f"Match Micro! P={P_test}, N={N_test}, tp={tp}, fp={fp}, score={score}")
                score2 = get_f1(P_test, N_test, tp, fp)
                if abs(score2 - 99.62264) < 0.001:
                    print(f"Match Macro! P={P_test}, N={N_test}, tp={tp}, fp={fp}, score={score2}")
