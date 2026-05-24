for y in range(1, 50000):
    x = round(y * 0.9881183)
    score = x / y * 100
    if abs(score - 98.81183) < 0.000005:
        print(f"Match Fraction! {x}/{y} = {score:.7f}")
