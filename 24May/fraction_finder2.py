scores = [58.51606, 73.58491, 76.2264, 0.37736, 1.13208]
for score in scores:
    print(f"--- Score: {score} ---")
    for den in range(1, 2000):
        for num in range(0, den + 1):
            val = num / den * 100
            diff = abs(val - score)
            if diff < 1e-5:
                print(f"Match: {num} / {den} = {val}")

    print("Checking F1 formulas:")
    # F1 = 2*TP / (P + Actual)
    # 58.51606 = 2*TP / (266 + 265)
    # TP = 58.51606 / 100 * 531 / 2
    for actual in range(200, 300):
        for tp in range(0, min(actual, 339)):
            # F1
            f1 = 2 * tp / (266 + actual) * 100
            if abs(f1 - score) < 1e-4:
                print(f"F1 Match: Actual={actual}, TP={tp}, F1={f1}")
            
            # Recall
            recall = tp / actual * 100
            if abs(recall - score) < 1e-4:
                print(f"Recall Match: Actual={actual}, TP={tp}, Recall={recall}")
            
            # Accuracy
            # tp + tn = tp + (339 - actual - fp) = tp + 339 - actual - (266 - tp) = 2*tp + 73 - actual
            acc = (2 * tp + 73 - actual) / 339 * 100
            if abs(acc - score) < 1e-4:
                print(f"Accuracy Match: Actual={actual}, TP={tp}, Acc={acc}")
