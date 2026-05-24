for tp in range(300):
    for fp in range(300):
        score = 100 * tp / 265 - 100 * fp / 74
        if abs(score - 81.61142) < 0.001:
            print(f"Match without C! TP={tp}, FP={fp}, Score={score}")
            
        score_with_c = 100 * tp / 265 - 100 * fp / 74 + 100
        if abs(score_with_c - 81.61142) < 0.001:
            print(f"Match with C=100! TP={tp}, FP={fp}, Score={score_with_c}")
