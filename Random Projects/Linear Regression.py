from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random
import random_data_generator as rdg

style.use('fivethirtyeight')

# dtype=np.float64 is needed to make sure that the mean stays accurate ex. int(7/2) = 3 , float(7/2) = 3.5

# X = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
# y = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)

"""
def create_dataset(points, variance, step, correlation=False):
    val = 1
    yl = []
    for i in range(points):
        ye = val + random.randrange(-variance, variance)
        yl.append(ye)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    Xl = [i for i in range(len(yl))]
    X = np.array(Xl, dtype=np.float64)
    y = np.array(yl, dtype=np.float64)
    return X, y
"""


def best_fit_slope_and_intercept(X, y):
    m = ((mean(X) * mean(y)) - mean(X*y)) / (np.square(mean(X)) - mean(np.square(X)))
    b = mean(y) - m * mean(X)
    return m, b


def square_error(y_point, y_line):
    se = sum(np.square(y_line - y_point))
    return se


def coefficient_of_determination(y_point, y_line):
    y_mean_line = [mean(y_point) for y in y_point]
    squared_error_regression = square_error(y_point, y_line)
    square_error_y_mean = square_error(y_point, y_mean_line)
    r_s = 1 - (squared_error_regression / square_error_y_mean)
    return r_s


# X, y = create_dataset(50, 50, 2, correlation='pos')
rdg.main()

m, b = best_fit_slope_and_intercept(X, y)

"""
for x in X:
    regression_line.append((m*x)+b)
"""

regression_line = [(m*x)+b for x in X]

r_squared = coefficient_of_determination(y, regression_line)
print(r_squared)

predict_x = 8
predict_y = (m*predict_x)+b

plt.scatter(X, y, color='r')
plt.plot(X, regression_line)
plt.scatter(predict_x, predict_y, s=100, color='g')
plt.show()
