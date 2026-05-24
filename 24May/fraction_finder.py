scores = [58.51606, 73.58491, 76.2264, 0.37736, 1.13208]
for score in scores:
    print(f"--- Score: {score} ---")
    best_diff = 1000
    best_frac = (0, 0)
    for den in range(1, 1000):
        for num in range(0, den + 1):
            val = num / den * 100
            diff = abs(val - score)
            if diff < 1e-4:
                print(f"Match: {num} / {den} = {val}")
