for tp in range(0, 339):
    for fp in range(0, 339):
        score = tp / 265 * 100 - fp / 74 * 100
        if abs(score - 81.61142) < 0.001:
            print(f"Match C=0! TP={tp}, FP={fp}, Score={score:.5f}")
            
        score_100 = tp / 265 * 100 - fp / 74 * 100 + 100
        if abs(score_100 - 81.61142) < 0.001:
            print(f"Match C=100! TP={tp}, FP={fp}, Score={score_100:.5f}")
