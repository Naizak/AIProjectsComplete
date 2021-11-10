import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')


def create_dataset(points, variance, step, correlation):
    val = 1
    yl = []
    for i in range(points):
        t = random.randrange(-variance, variance)
        ye = val + t
        yl.append(ye)
        if correlation == 1:
            val += step
        elif correlation == 0:
            val -= step
    Xl = [i for i in range(len(yl))]
    X = np.array(Xl)
    y = np.array(yl)
    return X, y


def random_input():
    a = int(random.random() * 1000)
    b = int(random.random() * 1000)
    if a > b:
        rnum = abs(random.randint(b, a))
    else:
        rnum = abs(random.randint(a, b))
    p = random.randrange(rnum)
    v = random.randrange(rnum)
    s = sigmoid(rnum/1000)
    if rnum % 2 == 0:
        c = 1
    else:
        c = 0
    return p, v, s, c


def sigmoid(x):
    sig = 1 / (1 + np.exp(-x))
    return sig


def main():
    # debuggin for i in range(100):
    points, variance, step, correlation = random_input()
    print("Amount of points:", points, '\n',
          "Amount of variance", variance, '\n',
          "Step amount", step, '\n',
          "correlation", correlation)
    X, y = create_dataset(points, variance, step, correlation)
    plt.scatter(X, y, c='k')  # random color np.random.rand(3,)
    plt.show()


main()
