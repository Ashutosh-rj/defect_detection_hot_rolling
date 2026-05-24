for y in range(1, 350):
    for x in range(1, y+1):
        if abs(x/y - 0.9962264) < 0.00001:
            print(f"{x}/{y} = {x/y:.7f}")
