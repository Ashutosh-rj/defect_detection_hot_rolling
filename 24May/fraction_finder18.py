import numpy as np

# Score * 265
val_202 = 60.29577 * 2.65
val_209 = 59.47986 * 2.65
val_267 = 39.87761 * 2.65

print(f"Target vals: {val_202}, {val_209}, {val_267}")

# Search for w
for w_int in range(1, 10000):
    w = w_int / 1000.0
    # x - 202w + xw = val_202 => x(1+w) = val_202 + 202w => x = (val_202 + 202w) / (1+w)
    x = (val_202 + 202 * w) / (1 + w)
    y = (val_209 + 209 * w) / (1 + w)
    z = (val_267 + 267 * w) / (1 + w)
    
    if abs(x - round(x)) < 1e-2 and abs(y - round(y)) < 1e-2 and abs(z - round(z)) < 1e-2:
        print(f"Match found! w={w}, x={round(x)}, y={round(y)}, z={round(z)}")

# What if it's (TP - w * FP) / Actual ?
# What if it's F1-like penalty?
