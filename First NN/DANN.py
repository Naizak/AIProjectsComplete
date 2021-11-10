# predicting a video games success in copies sold

"""
inputs/features:
    developer
    publisher
    critic score
    genre
    price

hidden layers (hl):
    critic score by genre
    critic score publisher
    critic score developer

output:
    success prediction (1/0)

feature set:
    shooters
        Call of Duty: Black Ops 4 = 14167945
        Splatoon 2 = 7037733
        Battlefield 5 = 3771595
        Overwatch = 4536011
        Fortnite = 1459012
        Destiny 2 = 4139803
    action
        Spider-Man = 8757859
        God of War = 6153627
        Red Dead Redemption 2 = 19711350
        Assassin's Creed Origin = 6537936
        Starlink Battle for Atlas = 567755
        Bayonetta 2 = 573643
    rpg
        Pokemon Let's Go Pikachu/Eevee! = 7294766
        Pokemon Ultra Sun/Ultra Moon = 7151768
        Fallout 76 = 2253016
        Fallout 4 = 8480129
        Skyrim = 1153690
        Kingdom Hearts 1.5 + 2.5 = 1750665

"""

import numpy as np
import activation_functions as litaf

feature_set = np.array([
    [1, 1, 83, 1],
    [2, 2, 83, 1],
    [3, 3, 73, 1],
    [4, 4, 90, 1],
    [5, 5, 78, 1],
    [6, 6, 83, 1],
    [7, 7, 87, 2],
    [8, 7, 94, 2],
    [9, 8, 97, 2],
    [10, 9, 81, 2],
    [10, 9, 70, 2],
    [11, 2, 91, 2],
    [12, 10, 79, 3],
    [12, 10, 84, 3],
    [13, 11, 49, 3],
    [13, 11, 84, 3],
    [13, 11, 84, 3],
    [14, 12, 84, 3]
])/100

labels = np.array(
    [
        14167945,
        7037733,
        3771595,
        4536011,
        1459012,
        4139803,
        8757859,
        6153627,
        19711350,
        6537936,
        567755,
        573643,
        7294766,
        7151768,
        2253016,
        8480129,
        1153690,
        1750665
     ]
)/100000000
labels = labels.reshape(18, 1)

np.random.seed(9)
alpha = .0001

# layer 1
l1w = np.random.rand(4, 3)

# layer 2
l2w = np.random.rand(3, 1)

for epoch in range(500000):

    # feed forward
    l1XW = np.dot(feature_set, l1w)
    l1z = litaf.sigmoid(l1XW)
    l2XW = np.dot(l1z, l2w)
    l2z = litaf.sigmoid(l2XW)

    # backprop step 1
    error = np.square(l2z - labels)
    print(error.sum())

    # backprop step 2
    l2_dcost_dpred = error
    l2_dpred_dz = litaf.sigmoid_der(l2z)
    l2_dz_dw = l1z
    l2_dcost_d2W = np.dot(l2_dz_dw.T, l2_dcost_dpred * l2_dpred_dz)

    l1_dcost_dz = l2_dcost_dpred * l2_dpred_dz
    l2_dz_dl1z = l2w

    l1_dcost_dl1z = np.dot(l1_dcost_dz, l2_dz_dl1z.T)
    l1_dl1z_dl1XW = litaf.sigmoid_der(l1XW)
    l1_dl1XW_d1W = feature_set
    l1_dcost_d1W = np.dot(l1_dl1XW_d1W.T, l1_dcost_dl1z * l1_dl1z_dl1XW)

    # update weights
    l1w -= alpha * l1_dcost_d1W
    l2w -= alpha * l2_dcost_d2W


test_example = np.array([.2, .2, .95, .01])
l1_answer = litaf.sigmoid(np.dot(test_example, l1w))
prediction = litaf.sigmoid(np.dot(l1_answer, l2w))
print(prediction)
