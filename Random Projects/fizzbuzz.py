import numpy as np
import tensorflow as tf

"""
We want the input to be a number, and the output to be the correct "fizzbuzz" 
representation of that number. 
In particular, we need to turn each input into a vector of "activations". 
One simple way would be to convert it to binary.
"""


def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])


"""
And our output will be a one-hot encoding of the fizzbuzz representation of the 
number, where the first position indicates "print as-is", the second indicates 
"fizz", and so on:
"""


def fizz_buzz_encode(i):
    if i % 15 == 0:
        return np.array([0, 0, 0, 1])
    elif i % 5 == 0:
        return np.array([0, 0, 1, 0])
    elif i % 3 == 0:
        return np.array([0, 1, 0, 0])
    else:
        return np.array([1, 0, 0, 0])


"""
Now we need to generate some training data. 
It would be cheating to use the numbers 1 to 100 in our training data, 
so let's train it on all the remaining numbers up to 1024:
"""

NUM_DIGITS = 10
trX = np.array([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
trY = np.array([fizz_buzz_encode(i) for i in range(101, 2 ** NUM_DIGITS)])

"""
Now we need to set up our model in tensorflow. 
Off the top of my head I'm not sure how many hidden units to use, maybe 10?
Yeah, possibly 100 is better. We can always change it later.
"""

NUM_HIDDEN = 100

"""
We'll need an input variable with width NUM_DIGITS, and an output variable with width 4:
"""

X = tf.placeholder("float", [None, NUM_DIGITS])
Y = tf.placeholder("float", [None, 4])

"""
Two layers, one hidden layer and one output layer.
Let's use randomly-initialized weights for our neurons:
"""


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


w_h = init_weights([NUM_DIGITS, NUM_HIDDEN])
w_o = init_weights([NUM_HIDDEN, 4])

"""
And we're ready to define the model. 
As I said before, one hidden layer, and let's use, I don't know, ReLU activation:
"""


def model(X, w_h, w_o):
    h = tf.nn.relu(tf.matmul(X, w_h))
    return tf.matmul(h, w_o)


"""
We can use softmax cross-entropy as our cost function and try to minimize it:
"""

py_x = model(X, w_h, w_o)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

"""
And, of course, the prediction will just be the largest output:
"""

predict_op = tf.argmax(py_x, 1)

"""
The predict_op function will output a number from 0 to 3, 
but we want a "fizz buzz" output:
"""


def fizz_buzz(i, prediction):
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]


"""
So now we're ready to train the model. 
Let's grab a tensorflow session and initialize the variables:
"""

with tf.Session() as sess:
    tf.initialize_all_variables().run()

"""
Now let's run, say, 1000 epochs of training?
Maybe that's not enough -- so let's do 10000 just to be safe.
And our training data are sequential, which I don't like, 
so let's shuffle them each iteration:

"""

for epoch in range(10000):
        p = np.random.permutation(range(len(trX)))
        trX, trY = trX[p], trY[p]

"""
And each epoch we'll train in batches of, I don't know, 128 inputs?
"""

BATCH_SIZE = 128

"""
So each training pass looks like
"""

for start in range(0, len(trX), BATCH_SIZE):
    end = start + BATCH_SIZE
    sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

"""
and then we can print the accuracy on the training data, since why not?
I find it helpful to see how the training accuracy evolves.
"""

print(epoch, np.mean(np.argmax(trY, axis=1) == sess.run(predict_op, feed_dict={X: trX, Y: trY})))

"""
So, once the model has been trained, it's fizz buzz time. 
Our input should just be the binary encoding of the numbers 1 to 100:
"""

numbers = np.arange(1, 101)
teX = np.transpose(binary_encode(numbers, NUM_DIGITS))

"""
And then our output is just our fizz_buzz function applied to the model output:
"""

teY = sess.run(predict_op, feed_dict={X: teX})
output = np.vectorize(fizz_buzz)(numbers, teY)

print(output)


