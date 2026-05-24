from fractions import Fraction

val = 0.5851606
f = Fraction(val).limit_denominator(100000)
print("Continued fraction limit 100000:", f)
f2 = Fraction(val).limit_denominator(1000)
print("Continued fraction limit 1000:", f2)

for den in range(1, 10000):
    num = round(val * den)
    if abs(num / den - val) < 1e-6:
        print(f"Match: {num} / {den} = {num/den}")
