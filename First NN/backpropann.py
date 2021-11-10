from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import activation_functions as litaf



np.random.seed(0)
input_into_nn, expected_answers = datasets.make_moons(100, noise=0.10)
plt.figure(figsize=(10, 7))
plt.scatter(input_into_nn[:, 0], input_into_nn[:, 1], c=expected_answers, cmap=plt.cm.winter)

expected_answers = expected_answers.reshape(100, 1)

weights_going_from_input_to_hidden_layer = np.random.rand(len(input_into_nn[0]), 4)
weights_going_from_hidden_to_output_layer = np.random.rand(4, 1)
learning_rate = 1

for epoch in range(200000):

    # Feed-forward
    raw_hidden_layer_answers = np.dot(input_into_nn, weights_going_from_input_to_hidden_layer)
    squishified_hidden_layer_answers = litaf.sigmoid(raw_hidden_layer_answers)

    raw_output_layer_answers = np.dot(squishified_hidden_layer_answers, weights_going_from_hidden_to_output_layer)
    squishified_output_layer_answers = litaf.sigmoid(raw_output_layer_answers)
    nn_prediction = squishified_output_layer_answers

    # Back-propagation Phase 1
    error_of_prediction = ((1/2) * (np.square(nn_prediction - expected_answers)))
    print(error_of_prediction.sum())

    der_of_cost_function_wrt_der_of_nn_prediction = nn_prediction - expected_answers
    der_of_nn_prediction_wrt_der_of_raw_output_layer_answers = litaf.sigmoid_der(raw_output_layer_answers)
    der_of_raw_output_layer_answers_wrt_der_of_weights_going_from_hidden_to_output_layer = squishified_hidden_layer_answers

    der_of_cost_function_wrt_der_of_weights_going_from_hidden_to_output_layer = np.dot(der_of_raw_output_layer_answers_wrt_der_of_weights_going_from_hidden_to_output_layer.T, der_of_cost_function_wrt_der_of_nn_prediction * der_of_nn_prediction_wrt_der_of_raw_output_layer_answers)

    # Back-propagation Phase 2
    der_of_cost_function_wrt_der_of_raw_output_layer_answers = der_of_cost_function_wrt_der_of_nn_prediction * der_of_nn_prediction_wrt_der_of_raw_output_layer_answers
    der_of_raw_output_layer_answers_wrt_der_of_sqishified_hidden_layer_answers = weights_going_from_hidden_to_output_layer
    der_of_cost_function_wrt_der_of_sqishified_hidden_layer_answers = np.dot(der_of_cost_function_wrt_der_of_raw_output_layer_answers, der_of_raw_output_layer_answers_wrt_der_of_sqishified_hidden_layer_answers.T)
    der_of_squishified_hidden_layer_answers_wrt_raw_hidden_layer_answers = litaf.sigmoid_der(raw_hidden_layer_answers)
    der_of_raw_hidden_layer_answers_wrt_der_of_weights_going_from_input_to_hidden_layer = input_into_nn

    der_of_cost_function_wrt_der_of_weights_going_from_input_to_hidden_layer = np.dot(der_of_raw_hidden_layer_answers_wrt_der_of_weights_going_from_input_to_hidden_layer.T, der_of_squishified_hidden_layer_answers_wrt_raw_hidden_layer_answers * der_of_cost_function_wrt_der_of_sqishified_hidden_layer_answers)
    
    weights_going_from_input_to_hidden_layer -= learning_rate * der_of_cost_function_wrt_der_of_weights_going_from_input_to_hidden_layer
    weights_going_from_hidden_to_output_layer -= learning_rate * der_of_cost_function_wrt_der_of_weights_going_from_hidden_to_output_layer



