for tp in range(300):
    for fp in range(300):
        score = 50 * tp / 265 - 50 * fp / 74 + 50
        if abs(score - 81.61142) < 0.001:
            print(f"Match BAcc! TP={tp}, FP={fp}, Score={score}")
