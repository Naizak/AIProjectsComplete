import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings

style.use('fivethirtyeight')

# plot1 = [1, 3]
# plot2 = [2, 5]
# euclidean_distance = np.sqrt(np.square(plot1[0] - plot2[0]) + np.square(plot1[1] - plot2[1]))

# where snakes go to learn new words
dataset = {'b': [[1, 2], [2, 3], [3, 1]], 'r': [[6, 5], [7, 7], [8, 6]]}
new_features = [3, 4]


def k_nearest_neighbors(data, predict, k):

    if len(data) >= k:
        warnings.warn("K is set to a value less than the total voting groups.")
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result


result = k_nearest_neighbors(dataset, new_features, 3)
print("New point is in the", "'", result, "'", "group.")

[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0], new_features[1], color=result)
plt.show()
