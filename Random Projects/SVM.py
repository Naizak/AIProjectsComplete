"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

class Support_Vector_Machine:
    def __init__(self, visulation=True):
        self.visulation = visulation
        self.colors = {1: 'r', -1: 'b'}
        if self.visulation:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)
    #train
    def fit(self, data):
        self.data = data
        # {||w||: [w, b]}
        opt_dict = {}
        transforms = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        # support vectors yi(xi.w+b) = 1

        step_size = [self.max_feature_value * 0.1,
                     self.max_feature_value * 0.01,
                     # point of expense:
                     self.max_feature_value * 0.001]

        # extremely expensive
        b_range_multiple = 5
        # we don't need to take as small of steps
        # with b as we do with w
        b_multiple = 5
        latest_optimum = self.max_feature_value*10

        for steps in step_size:
            w = np.array([latest_optimum, latest_optimum])
            # we can do this because convex
            optimized = False
            while not optimized:
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple),
                                   self.max_feature_value*b_range_multiple,
                                   steps*b_multiple):

                    for transformation in transforms:
                        w_t = w*transformation
                        found_option = True
                        # weakest link in the SVM fundamentally
                        # SMO attemps to fix this a bit
                        # yi(xi.w+b) >= 1
                        #
                        # #### add a break here later..

                        for i in self.data:

                            for xi in self.data[i]:
                                yi = i
                                if not yi*(np.dot(w_t, xi)+b) >= 1:
                                    found_option = False


                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]



            print("out of the loop")
            if w[0] < 0:
                optimized = True
                print('Optimized a step.')
            else:
                w = w-steps


        norms = sorted([n for n in opt_dict])
        # ||w|| : [w, b]
        opt_choice = opt_dict[norms[0]]
        self.w = opt_choice[0]
        self.b = opt_choice[1]
        latest_optimum = opt_choice[0][0]+steps*2

    def predict(self, features):
        # sign(x.w+b)
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        if classification != 0 and self.visulation:
            self.ax.scatter(features[0], features[1], s=200,
                            marker='*', c=self.colors[classification])
        return classification

    def visualize(self):
        [[self.ax.scatter(x[0], x[1], s=100, color=self.colors[i])
          for x in data_dict[i]] for i in data_dict]
        # hyperplane = x.w+b
        # v = x.w+b
        # psv = 1
        # nsv = -1
        # dec = 0

        def hyperplane(x, w, b, v):
            h = (-w[0]*x-b+v) / w[1]
            return h

        datarange = (self.min_feature_value*0.9, self.max_feature_value*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # (w.x+b) = 1
        # positive support vector hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2])

        # (w.x+b) = -1
        # negative support vector hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2])

        # (w.x+b) = 0
        # decision boundary vector hyperplane
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2])
        plt.show()


data_dict = {-1: np.array([[1, 7], [2, 8], [3, 8]]),
             1: np.array([[5, 1], [6, -1], [7, 3]])}

svm = Support_Vector_Machine()
svm.fit(data=data_dict)
svm.visualize()
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets._samples_generator import make_blobs

np.random.seed(6)

(X, y) = make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)
# we need to add 1 to X values (we can say it's a bias)
X1 = np.c_[np.ones((X.shape[0])), X]

plt.scatter(X1[:, 1], X1[:, 2], marker='o', c=y)
plt.axis([-5, 10, -12, -1])
plt.show()

positiveX = []
negativeX = []

for i, v in enumerate(y):
    if v == 0:
        negativeX.append(X[i])
    else:
        positiveX.append(X[i])

# our data dictionary
data_dict = {-1: np.array(negativeX), 1: np.array(positiveX)}

# all the required variables
w = []  # weights 2d vector
b = []  # bias

max_feature_value = float('-inf')
min_feature_value = float('inf')

# finding max & min feature values
for yi in data_dict:
    if np.amax(data_dict[yi]) > max_feature_value:
        max_feature_value = np.amax(data_dict[yi])
    if np.amin(data_dict[yi]) < min_feature_value:
        min_feature_value = np.amin(data_dict[yi])

learning_rate = [max_feature_value * 0.1, max_feature_value * 0.01, max_feature_value * 0.001]


def svm_training(data_dict):
    i = 1
    global w
    global b
    # {||w||: [w,b]}
    # A dict where the eucledian distance found from the np.linalg.norm(w_t) is the key and the vector
    # w is [w_optimum, w_optimum]* transforms = [[1, 1], [-1, 1], [1, -1], [-1, -1]] and
    # b is the index from the range ((-1*(max_feature_value*b_step_size),max_feature_value*b_step_size)) as the bias
    # are the values
    length_w_vector = {}
    transforms = [[1, 1], [-1, 1], [1, -1], [-1, -1]]

    b_step_size = 2
    b_multiple = 5
    w_optimum = max_feature_value * 0.5

    for alpha in learning_rate:
        w = np.array([w_optimum, w_optimum])
        optimized = False
        while not optimized:
            # b = [-maxvalue to maxvalue] we wanna maximize the b values so check for every b value
            for b in np.arange(-1*(max_feature_value*b_step_size), max_feature_value*b_step_size, alpha*b_multiple):
                for transformation in transforms:  # transforms = [[1, 1], [-1, 1], [1, -1], [-1, -1]]
                    w_t = w * transformation
                    correctly_classified = True
                    # every data point should be correct
                    for yi in data_dict:
                        for xi in data_dict[yi]:
                            # Why are we dotting the w_t vector with every point then adding the bias then multiplying
                            # it by the group label (-1, or 1) and seeing if this result is less than 1?
                            if yi*(np.dot(w_t, xi)+b) < 1: # we want yi*(np.dot(w_t,xi)+b)>=1 for correct classification
                                correctly_classified = False
                    if correctly_classified:
                        length_w_vector[np.linalg.norm(w_t)] = [w_t, b]  # store w, b for minimum magnitude
            if w[0] < 0:
                optimized = True
            else:
                w = w - alpha
        norms = sorted([n for n in length_w_vector])
        # minimum_w_length is to equal the values of the length_w_vector{} at key norms[0]
        minimum_w_length = length_w_vector[norms[0]]
        w = minimum_w_length[0]
        b = minimum_w_length[1]

        w_optimum = w[0] + alpha*2


svm_training(data_dict)

colors = {1: 'r', -1: 'b'}
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)


def visualize(data_dict):
    # [[ax.scatter(x[0],x[1],s=100,color=colors[i]) for x in data_dict[i]] for i in data_dict]
    plt.scatter(X1[:, 1], X1[:, 2], marker='o', c=y)
    # hyperplane = x.w+b
    # v = x.w+b
    # psv = 1
    # nsv = -1
    # dec = 0

    def hyperplane_value(x, w, b, v):
        h_value = (-w[0]*x-b+v)/w[1]
        return h_value

    datarange = (min_feature_value*0.9, max_feature_value*1)
    hyp_x_min = datarange[0]
    hyp_x_max = datarange[1]

    # (w.x+b) = 1
    # positive support vector hyperplane
    psv1 = hyperplane_value(hyp_x_min, w, b, 1)
    psv2 = hyperplane_value(hyp_x_max, w, b, 1)
    ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k')

    # (w.x+b) = -1
    # negative support vector hyperplane
    nsv1 = hyperplane_value(hyp_x_min, w, b, -1)
    nsv2 = hyperplane_value(hyp_x_max, w, b, -1)
    ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')

    # (w.x+b) = 0
    # decision boundary
    db1 = hyperplane_value(hyp_x_min, w, b, 0)
    db2 = hyperplane_value(hyp_x_max, w, b, 0)
    ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')

    plt.axis([-5, -10, -12, -1])
    plt.show()


visualize(data_dict)


def predict(features):
    # sign(x.w+b)
    dot_result = np.sign(np.dot(np.array(features), w)+b)
    int_dot_result = dot_result.astype(int)
    return int_dot_result


for i in X[:5]:
    print(predict(i), end=', ')


"""
l = []
for xi in X:
    l.append(predict(xi[:6]))
l = np.array(l).astype(int)
print(l)
"""