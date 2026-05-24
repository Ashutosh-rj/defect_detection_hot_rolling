for tp in range(0, 266):
    for tn in range(0, 75):
        score = (tp / 265 + tn / 74) / 2 * 100
        if abs(score - 81.61142) < 0.0001:
            print(f"Match base! TP={tp}, TN={tn}, FP={74-tn}, FN={265-tp}, Score={score:.5f}")
            print(f"Total Predicted Positive = TP+FP = {tp + (74-tn)}")
