import random
import csv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


with open('randpct.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(30000):
        r = random.random()
        n = "%.6f" % r
        writer.writerow([n])


# # --- Data Structuring ---

# getting path of data
path = "C:\\Users\\naiza\\PycharmProjects\\AIProjects\\randpct.csv"

# reading path of data into program
df = pd.read_csv(path)

# taking only the meaningful features
df = df[['X', 'y']]

A = np.array(df['X'])
# print(A)
b = np.array(df['y'])
# print(b)


X = A[:100]
# print(X)
y = b[:100]
# print(y)

y = [int(np.floor(e * 100)) for e in y]
# print(y)

y_odd = [i for i in y if i % 2 != 0]
y_set = list(set(y_odd))
y_prime = []

for i in y_set:
    for num in range(2, i):
        if(num % i) != 0:
            y_prime.append(i)

y_prime = list(set(y_prime))

print(y_set)
print(y_odd)
print(y_prime)


plt.scatter(X, y, c='k')

plt.plot(X, y, 'ko-', linewidth=2, markersize=5)
plt.plot(X, y, color='black', linestyle='solid', marker='o',
     markerfacecolor='blue', markersize=5)
plt.show()