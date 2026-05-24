import numpy as np
from sklearn.metrics import *

P = 265
N = 74
TP = 227
FP = 4
FN = P - TP
TN = N - FP

y_true = np.array([1]*P + [0]*N)
y_pred = np.array([1]*TP + [0]*FN + [1]*FP + [0]*TN)

print("Accuracy:", accuracy_score(y_true, y_pred) * 100)
print("Balanced Accuracy:", balanced_accuracy_score(y_true, y_pred) * 100)
print("F1 Macro:", f1_score(y_true, y_pred, average='macro') * 100)
print("F1 Micro:", f1_score(y_true, y_pred, average='micro') * 100)
print("F1 Weighted:", f1_score(y_true, y_pred, average='weighted') * 100)
print("F1 Binary:", f1_score(y_true, y_pred) * 100)
print("Precision:", precision_score(y_true, y_pred) * 100)
print("Recall:", recall_score(y_true, y_pred) * 100)
print("Jaccard:", jaccard_score(y_true, y_pred) * 100)
print("Matthew:", matthews_corrcoef(y_true, y_pred) * 100)

