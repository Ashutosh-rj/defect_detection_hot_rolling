scores = [77.67976, 79.18919, 79.34727, 80.10199, 80.85671]

for score in scores:
    val = score / 100
    for den in range(1, 10000):
        num = round(val * den)
        if abs(num / den - val) < 1e-6:
            print(f"Score {score} matches {num}/{den}")
            break
