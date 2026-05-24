for tp in range(232):
    fp = 231 - tp
    fn = 265 - tp
    score = 100 - fn * 0.37735849 - fp * 0.594085
    if abs(score - 81.61142) < 0.1:
        print(f"Match new penalty! TP={tp}, FP={fp}, Score={score}")
