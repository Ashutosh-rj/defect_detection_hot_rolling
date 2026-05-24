for P in range(100, 339):
    N = 339 - P
    for tp in range(100, min(P, 231) + 1):
        fp = 231 - tp
        if fp > N: continue
        tn = N - fp
        score = (tp / P + tn / N) / 2 * 100
        if abs(score - 81.61142) < 0.0001:
            print(f"Match! P={P}, N={N}, TP={tp}, FP={fp}, TN={tn}, FN={P-tp}, Score={score:.5f}")
