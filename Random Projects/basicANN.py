from matplotlib import pyplot as plt
import numpy as np

# input = np.linspace(-10, 10, 100)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))


"""
sigmoid_plot = plt.plot(input, sigmoid(input), c="b")
plt.show(sigmoid_plot)
"""

feature_set = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]])
labels = np.array([1, 0, 0, 1, 1])
labels = labels.reshape(5, 1)  # Transpose of labels

np.random.seed(42)
weights = np.random.rand(3, 1)
bias = np.random.rand(1)
alpha = 0.05


for epoch in range(200000):
    inputs = feature_set

    # feedforward step 1
    XW = np.dot(feature_set, weights) + bias

    # feedforward step 2
    z = sigmoid(XW)

    # backpropagation step 1
    error = z - labels
    print(error.sum())

    # backpropagation step 2
    dcost_dpred = error
    dpred_dz = sigmoid_der(z)
    z_delta = dcost_dpred * dpred_dz
    weights -= alpha * np.dot(feature_set.T, z_delta)

    for num in z_delta:
        bias -= alpha * num

single_point = np.array([1, 1, 0])
result = sigmoid(np.dot(single_point, weights) + bias)
print(result)