"""
Naive Bayes Algorithm Tutorial
This tutorial is broken down into the following steps:

Handle Data: Load the data from CSV file and split it into training and test datasets.
Summarize Data: summarize the properties in the training dataset so that we can calculate probabilities and make predictions.
Make a Prediction: Use the summaries of the dataset to generate a single prediction.
Make Predictions: Generate predictions given a test dataset and a summarized training dataset.
Evaluate Accuracy: Evaluate the accuracy of predictions made for a test dataset as the percentage correct out of all predictions made.
Tie it Together: Use all of the code elements to present a complete and standalone implementation of the Naive Bayes algorithm.
"""

import csv
import random
import math


"""
1. Handle Data
The first thing we need to do is load our data file. The data is in CSV format without a header line or any quotes. 
We can open the file with the open function and read the data lines using the reader function in the csv module.
We also need to convert the attributes that were loaded as strings into numbers that we can work with them. 
Below is the load_csv() function for loading the Pima indians dataset.
"""


def load_csv(filename):
    lines = csv.reader(open(filename, 'r'))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset


# We can test this function by loading the pima indians dataset and printing the number of data instances that were loaded.

"""
filename = 'pima-indians-diabetes.data.csv'
dataset = load_csv(filename)
print('Loaded data file', filename, 'with', len(dataset), 'rows')
"""

"""
Next we need to split the data into a training dataset that Naive Bayes can use to make predictions and a test 
dataset that we can use to evaluate the accuracy of the model. We need to split the data set randomly into train 
and datasets with a ratio of 67% train and 33% test (this is a common ratio for testing an algorithm on a dataset).
Below is the split_dataset() function that will split a given dataset into a given split ratio.
"""

def split_dataset(dataset, split_ratio):
    train_size = int(len(dataset)*split_ratio)
    train_set = []
    copy = list(dataset)
    while len(train_set) < train_size:
        index = random.randrange(len(copy))
        train_set.append(copy.pop(index))
    return [train_set, copy]


# We can test this out by defining a mock dataset with 5 instances, split it into training and testing datasets and
# print them out to see which data instances ended up where.
"""
dataset = [[1], [2], [3], [4], [5]]
splitRatio = 0.67
train, test = split_dataset(dataset, splitRatio)
print('Split', len(dataset), 'rows into train with', train, 'and test with', test)
"""

"""
2. Summarize Data
The naive bayes model is comprised of a summary of the data in the training dataset. 
This summary is then used when making predictions.

The summary of the training data collected involves the mean and the standard deviation for each attribute, by class 
value. For example, if there are two class values and 7 numerical attributes, then we need a mean and standard deviation
for each attribute (7) and class value (2) combination, that is 14 attribute summaries.

These are required when making predictions to calculate the probability of specific attribute values belonging to each class value.

We can break the preparation of this summary data down into the following sub-tasks:

Separate Data By Class
Calculate Mean
Calculate Standard Deviation
Summarize Dataset
Summarize Attributes By Class
Separate Data By Class

The first task is to separate the training dataset instances by class value so that we can calculate statistics for each
class. We can do that by creating a map of each class value to a list of instances that belong to that class and sort 
the entire dataset of instances into the appropriate lists.

The separate_by_class() function below does just this.
"""


def separate_by_class(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        v = vector[-1]
        v2 = vector[-2]
        if vector[-1] not in separated:
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated


"""
You can see that the function assumes that the last attribute (-1) is the class value. 
The function returns a map of class values to lists of data instances.
"""

# We can test this function with some sample data, as follows:
"""
dataset = [[1, 20, 1], [2, 21, 0], [3, 22, 1]]
separated = separate_by_class(dataset)
print('Separated instances:', separated)
"""


"""
Calculate Mean
We need to calculate the mean of each attribute for a class value. The mean is the central middle or central tendency of 
the data, and we will use it as the middle of our gaussian distribution when calculating probabilities.

We also need to calculate the standard deviation of each attribute for a class value. The standard deviation describes
the variation of spread of the data, and we will use it to characterize the expected spread of each attribute in our 
Gaussian distribution when calculating probabilities.

The standard deviation is calculated as the square root of the variance. 
The variance is calculated as the average of the squared differences for each attribute value from the mean. 
Note we are using the N-1 method, which subtracts 1 from the number of attribute values when calculating the variance.
"""


def mean(numbers):
    mean = sum(numbers)/float(len(numbers))
    return mean


def standard_deviation(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg, 2) for x in numbers])/float(len(numbers)-1)
    stdev = math.sqrt(variance)
    return stdev


# We can test this by taking the mean of the numbers from 1 to 5.
"""
numbers = [1, 2, 3, 4, 5]
print('Summary of', numbers, ':', 'mean =', mean(numbers), 'standard_deviation =', standard_deviation(numbers))
"""


"""
Summarize Dataset
Now we have the tools to summarize a dataset. For a given list of instances (for a class value) we can calculate the 
mean and the standard deviation for each attribute.

The zip function groups the values for each attribute across our data instances into their own lists so that we can 
compute the mean and standard deviation values for the attribute.
"""


def summarize(dataset):
    summaries = [(mean(attribute), standard_deviation(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

# We can test this summarize() function with some test data that shows markedly different mean and standard deviation
# values for the first and second data attributes.

"""
dataset = [[1, 20, 0], [2, 21, 1], [3, 22, 0]]
summary = summarize(dataset)
print('Attribute summaries:', summary)
"""


"""
Summarize Attributes By Class
We can pull it all together by first separating our training dataset into instances grouped by class. 
Then calculate the summaries for each attribute.
"""

def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries

# We can test this summarizeByClass() function with a small test dataset.

"""
dataset = [[1, 20, 1], [2, 21, 0], [3, 22, 1], [4, 22, 0]]
summary = summarize_by_class(dataset)
print('Summary by class value:', summary)
"""

"""
3. Make Prediction
We are now ready to make predictions using the summaries prepared from our training data. 
Making predictions involves calculating the probability that a given data instance belongs to each class, 
then selecting the class with the largest probability as the prediction.

We can divide this part into the following tasks:

Calculate Gaussian Probability Density Function
Calculate Class Probabilities
Make a Prediction
Estimate Accuracy
Calculate Gaussian Probability Density Function

We can use a Gaussian function to estimate the probability of a given attribute value, 
given the known mean and standard deviation for the attribute estimated from the training data.

Given that the attribute summaries where prepared for each attribute and class value, 
the result is the conditional probability of a given attribute value given a class value.

See the references for the details of this equation for the Gaussian probability density function. 
In summary we are plugging our known details into the Gaussian (attribute value, mean and standard deviation) 
and reading off the likelihood that our attribute value belongs to the class.

In the calculate_probability() function we calculate the exponent first, then calculate the main division. 
This lets us fit the equation nicely on two lines.
"""


def calculate_probability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean, 2)/(2*math.pow(stdev, 2))))
    probability = (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
    return probability


# We can test this with some sample data, as follows.

"""
x = 71.5
mean = 73
stdev = 6.2
probability = calculate_probability(x, mean, stdev)
print('Probability of belonging to this class:', probability)
"""


"""
Calculate Class Probabilities
Now that we can calculate the probability of an attribute belonging to a class, we can combine the probabilities of all 
of the attribute values for a data instance and come up with a probability of the entire data instance belonging to the class.

We combine probabilities together by multiplying them. In the calculateClassProbabilities() below, the probability of a 
given data instance is calculated by multiplying together the attribute probabilities for each class. the result is a 
map of class values to probabilities.
"""


def calculate_class_probabilities(summaries, input_vector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = input_vector[i]
            probabilities[classValue] *= calculate_probability(x, mean, stdev)
    return probabilities

# We can test the calculate_class_probabilities() function.


"""
summaries = {0: [(1, 0.5)], 1: [(20, 5.0)]}
input_vector = [1.1, '?']
probabilities = calculate_class_probabilities(summaries, input_vector)
print('Probabilities for each class:', probabilities)
"""


"""
Make a Prediction
Now that can calculate the probability of a data instance belonging to each class value, we can look for the largest 
probability and return the associated class.
The predict() function belong does just that.
"""


def predict(summaries, input_vector):
    probabilities = calculate_class_probabilities(summaries, input_vector)
    best_label, best_prob = None, -1
    for classValue, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = classValue
    return best_label

# We can test the predict() function as follows:

"""
summaries = {'A': [(1, 0.5)], 'B': [(20, 5.0)]}
input_vector = [1.1, '?']
result = predict(summaries, input_vector)
print('Prediction:', result)
"""


"""
4. Make Predictions
Finally, we can estimate the accuracy of the model by making predictions for each data instance in our test dataset. 
The get_predictions() will do this and return a list of predictions for each test instance.
"""


def get_predictions(summaries, test_set):
    predictions = []
    for i in range(len(test_set)):
        result = predict(summaries, test_set[i])
        predictions.append(result)
    return predictions

# We can test the getPredictions() function.

"""
summaries = {'A': [(1, 0.5)], 'B': [(20, 5.0)]}
test_set = [[1.1, '?'], [19.1, '?']]
predictions = get_predictions(summaries, test_set)
print('Predictions:', predictions)
"""

"""
5. Get Accuracy
The predictions can be compared to the class values in the test dataset and a classification accuracy can be 
calculated as an accuracy ratio between 0& and 100%. The get_accuracy() will calculate this accuracy ratio.
"""


def get_accuracy(test_set, predictions):
    correct = 0
    for x in range(len(test_set)):
        if test_set[x][-1] == predictions[x]:
            correct += 1
    accuracy = (correct/float(len(test_set))) * 100.0
    return accuracy


# We can test the get_accuracy() function using the sample code below.
"""
test_set = [[1, 1, 1, 'a'], [2, 2, 2, 'a'], [3, 3, 3, 'b']]
predictions = ['a', 'a', 'a']
accuracy = get_accuracy(test_set, predictions)
print('Accuracy:', accuracy)
"""

"""
6. Tie it Together
Finally, we need to tie it all together.
"""


def main():
    filename = 'pima-indians-diabetes.data.csv'
    split_ratio = 0.67
    dataset = load_csv(filename)
    training_set, test_set = split_dataset(dataset, split_ratio)
    print('Split', len(dataset), 'rows into train =', len(training_set), 'and test =', len(test_set), 'rows')
    # prepare model
    summaries = summarize_by_class(training_set)
    # test model
    predictions = get_predictions(summaries, test_set)
    accuracy = get_accuracy(test_set, predictions)
    print('Accuracy:', accuracy, '%')


main()
