import tensorflow as tf

# Example! neural network
n_features = 10
n_dense_neurons = 3
x = tf.placeholder(tf.float32, (None, n_features))
W = tf.Variable(tf.random_normal([n_features,n_dense_neurons]))
b = tf.Variable(tf.ones([n_dense_neurons]))
xW = tf.matmul(x,W)
z = tf.add(xW,b)
a = tf.sigmoid(z)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    layer_out = sess.run(a, feed_dict={x: np.random.random([1, n_features])})
print(layer_out)


# Neural network calculating y = mx + b
np.random.rand(2)
m = tf.Variable(0.44)
b = tf.Variable(0.87)
error = 0
for x,y in zip(x_data,y_label):
    y_hat = m*x + b
    error += (y-y_hat)**2
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    training_steps = 1
    for i in range(training_steps):
        sess.run(train)
    final_slope, final_intercept = sess.run([m,b])
x_test = np.linspace(-1,11,10)
y_pred_plot = final_slope*x_test + final_intercept
plt.plot(x_test, y_pred_plot)
plt.plot(x_data, y_label, '*')
plt.show()


# Neural network calculating big linear regression model
import tensorflow as tf
x_data = np.linspace(0.0,10.0,1000000)
noise = np.random.randn(len(x_data))
x_data
noise.shape
# y = mx + b | b = 5
y_true = (0.5 * x_data) + 5
y_true = (0.5 * x_data) + 5 + noise
x_df = pd.DataFrame(data=x_data,columns=['X Data'])
y_df = pd.DataFrame(data=y_true, columns=['Y'])
x_df.head()
y_df.head()
my_data = pd.concat([x_df,y_df], axis=1)
my_data
my_data.sample(n=250)
my_data.sample(n=250).plot(kind='scatter', x='X Data', y='Y')
# It is not possible to add 1000000 points at once to the model, so you need to feed it in batches
batch_size = 8
m = tf.Variable(0.5) # slope
np.random.randn(2)
m = tf.Variable(0.024) # slope
b = tf.Variable(-0.592)
# Create 2 placeholders. One for the X and one for the Y
xph = tf.placeholder(tf.float32,[batch_size])
yph = tf.placeholder(tf.float32,[batch_size])
y_model = m*xph
y_model = m*xph + b
error = tf.reduce_sum(tf.square(yph-y_model))
# Loss function uses tf.square for the conventience function
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init) # first initialize variables
with tf.Session() as sess:
    sess.run(init) # first initialize variables
    # Feeding 1000 batches of data. Each batch is 8 datapoints: x datapoint and y label
    batches = 1000
    for i in range(batches):
        # To make the data usefull this will create 8 random data points from the dataset
        # Creates random integer from 0 to len(x_data). Just filling in len(x_data) is a
        # shorthand. You grab 8 of those (batch_size)
        rand_ind = np.random.randint(len(x_data), size=batch_size)
        # Inputting random data points into x_data and y_true to create feed dictionary
        feed = {xph: x_data[rand_ind], yph: y_true[rand_ind]}
        sess.run(train, feed_dict = feed)
    model_m, model_b = sess.run([m,b])
model_m
model_b
y_hat = x_data * model_m + model_b
my_data.sample(250).plot(kind='scatter', x='X Data', y='Y')
plt.plot(x_data, y_hat, 'r')