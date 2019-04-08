# One of the simplest machine learning models It operates on a line
#   which is (y = mx + b)
# You have data points and try to find a line of best fit through those points
# Model optimizes m and b to find the best line until it minimizes loss
#   Loss is difference between actual y value and y value that the line would
#   predict. Loss is the the difference between actual y points and the y
#   points that run through the line itself.
# Training the model:
#   - take x values and expected y values as inputs
#   - Starting with placeholder noes for x and y inputs.
#   - At end of model only take x values as inputs and have y values as output.
#   - start with guess for m and b and measure loss
#   - Run program to adjust ma nd b to minize loss based on inputs
# Final program will be a good line through the data

import tensorflow as tf

# y = mx + b
# y = Wx + b <-- same thing. W for weight
# x = [1, 2, 3, 4]
# y = [0, -1, -2, -3]

# W is also called 'slope'
# b is also called 'intercept'

W = tf.Variable([-.5], dtype=tf.float32)  # probably a negative slope
b = tf.Variable([.5], dtype=tf.float32)  # probably a postive intercept

x = tf.placeholder(dtype=tf.float32)
y = tf.placeholder(dtype=tf.float32)

linear_model = (W * x) + b

# Loss is the squared tensorflow output minus the actual y that we wanted
# to see. It is the value that indicates how wrong we are compared to the
# actual output. < 5 is a very good value. > 30 is very bad
loss = tf.reduce_sum(tf.square(linear_model - y))

# This is the optimizer, modifying W and b by a value each iteration.
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

x_train = [1, 2, 3, 4]
# output: [0. 0.5 -1 -1.5]
y_train = [0, -1, -2, -3]

session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)
# res = session.run(linear_model, {
#     x: x_train
# })
# loss = session.run(loss, {
#     x: x_train,
#     y: y_train
# })
# print(res)
# print(loss)

for i in range(1000):
    session.run(train, {
        x: x_train,
        y: y_train
    })
