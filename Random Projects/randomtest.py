import random

variance = 80
r = set()

for i in range(100):
    rand = abs(random.randrange(-variance, variance))
    r.add(rand)
print(r)

