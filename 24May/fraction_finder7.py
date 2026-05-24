scores_and_p = [
    (0.37736, 1),
    (1.13208, 3),
    (73.58491, 200),
    (58.51606, 266)
]

print("Checking metrics for a consistent Actual value...")
for actual in range(1, 1000):
    for metric_type in ['Jaccard', 'Recall', 'F1', 'Precision', 'Accuracy']:
        valid = True
        tps = []
        for score, p in scores_and_p:
            match_tp = -1
            for tp in range(0, p + 1):
                val = -1
                if metric_type == 'Jaccard':
                    val = tp / (p + actual - tp) * 100
                elif metric_type == 'Recall':
                    val = tp / actual * 100
                elif metric_type == 'F1':
                    val = 2*tp / (p + actual) * 100
                elif metric_type == 'Precision':
                    val = tp / p * 100 if p > 0 else 0
                elif metric_type == 'Accuracy':
                    val = (tp + (339 - p - actual + tp)) / 339 * 100
                
                if abs(val - score) < 1e-3:
                    match_tp = tp
                    break
            
            if match_tp == -1:
                valid = False
                break
            else:
                tps.append(match_tp)
                
        if valid:
            print(f"BINGO! Metric: {metric_type}, Actual: {actual}, TPs: {tps}")

print("Done.")
