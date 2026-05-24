scores = {
    58.51606: 266, 
    73.58491: 200, 
    76.2264: 202, 
    0.37736: 1, 
    1.13208: 3
}

print("Checking F1 formulas:")
for score, P in scores.items():
    print(f"--- Score: {score}, P: {P} ---")
    for actual in range(50, 339):
        for tp in range(0, min(actual + 1, P + 1)):
            # F1
            f1 = 2 * tp / (P + actual) * 100
            if abs(f1 - score) < 1e-4:
                print(f"F1 Match: Actual={actual}, TP={tp}, P={P}, F1={f1}")
            
            # Recall
            recall = tp / actual * 100
            if abs(recall - score) < 1e-4:
                print(f"Recall Match: Actual={actual}, TP={tp}, P={P}, Recall={recall}")
            
            # Jaccard
            jaccard = tp / (P + actual - tp) * 100
            if abs(jaccard - score) < 1e-4:
                print(f"Jaccard Match: Actual={actual}, TP={tp}, P={P}, Jaccard={jaccard}")

