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


# Sample estimator API. The API is usefull for simpler TensorFlow tasks such as regression
# or classification.
import tensorflow as tf
# Create a list of feature columns. Every column has a special feature, such as a numeric column
# or a categorical column. You provide it with a key and shape.
feat_cols = [ tf.feature_column.numeric_column('x', shape][1] ]
feat_cols = [ tf.feature_column.numeric_column('x', shape[1]) ]
feat_cols = [ tf.feature_column.numeric_column('x', shape=[1]) ]
feat_cols
# Create a simple estimator.
estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)
# Do a train test split. This should always be done, regardless of using an estimator or not.
from sklearn.model_selection import train_test_split
x_train, x_eval, y_train, y_eval = train_test_split(x_data, y_true, test_size=0.3, random_state=101)
x_data = np.linspace(0.0,10.0,1000000)
import numpy as np
x_data = np.linspace(0.0,10.0,1000000)
noise = np.random.randn(len(x_data))
y_true = (0.5 * x_data) + 5 + noise
x_train, x_eval, y_train, y_eval = train_test_split(x_data, y_true, test_size=0.3, random_state=101)
# Create some input functions
input_func = tf.estimator.inputs.numpy_input_fn({'x': x_train}, y_train, batch_size=8, num_epochs=None, shuffle=True)
train_input_func = tf.estimator.inputs.numpy_input_fn({'x': x_train}, y_train, batch_size=8, num_epochs=1000, shuffle=False)
input_func = tf.estimator.inputs.numpy_input_fn({'x': x_eval}, y_eval, batch_size=8, num_epochs=None, shuffle=True)
estimator.train(input_fn=input_func, steps=1000)
# Train the data based of the training data function
train_metrics = estimator.evaluate(input_fn=train_input_func, steps=1000)
# Evaluate the training data metrics
eval_metrics = estimator.evaluate(input_fn=eval_input_func, steps=1000)
input_func = tf.estimator.inputs.numpy_input_fn({'x': x_train}, y_train, batch_size=8, num_epochs=None, shuffle=True)
train_input_func = tf.estimator.inputs.numpy_input_fn({'x': x_train}, y_train, batch_size=8, num_epochs=1000, shuffle=False)
input_func = tf.estimator.inputs.numpy_input_fn({'x': x_train}, y_train, batch_size=8, num_epochs=1000, shuffle=False)
estimator.train(input_fn=input_func, steps=1000)
train_metrics = estimator.evaluate(input_fn=train_input_func, steps=1000)
eval_metrics = estimator.evaluate(input_fn=eval_input_func, steps=1000)
%magic
%history -g eval
# Evaluate the evaluation metrics which reports back the loss function
eval_input_func = tf.estimator.inputs.numpy_input_fn({'x': x_eval}, y_eval, batch_size=8, num_epochs=1000, shuffle=False)
eval_metrics = estimator.evaluate(input_fn=eval_input_func, steps=1000)
print('Training data metrics')
print(train_metrics)
print('Eval metrics')
print(eval_metrics)
# If train_metrics and eval_metrics are very far apart, that means the data is overfitting. The evaluation loss to perform slightly worse then training data. In general you want these to be similar. 
# So what happens with new data, or predicting from new data points.
brand_new_data = np.linspace(0, 10, 10)
input_fn_predict = tf.estimator.inputs.numpy_input_fn({'x': brand_new_data}, shuffle=False)
estimator.predict(input_fn=input_fn_predict)
# this is a generator object to iterate over, or you can cast it as a list to see the predictions or iterate over it.
list(estimator.predict(input_fn=input_fn_predict))
predictions = []
for pred in estimator.predict(input_fn=input_fn_predict):
    predictions.append(pred['predictions'])
predictions
my_data.sample(n=250).plot(kind='scatter', x='X Data', y='Y')
%history -g my_data
my_data = pd.concat([x_df,y_df], axis=1)
import pandas as pd
%history -g x_df
x_df = pd.DataFrame(data=x_data,columns=['X Data'])
%history -g y_df
# Plot out the data
y_df = pd.DataFrame(data=y_true, columns=['Y'])
my_data = pd.concat([x_df,y_df], axis=1)
my_data.sample(n=250).plot(kind='scatter', x='X Data', y='Y')
%matplotlib
my_data.sample(n=250).plot(kind='scatter', x='X Data', y='Y')
plt.plot(brand_new_data, predictions, 'r')
import matplotlib.pyplot as plt
plt.plot(brand_new_data, predictions, 'r*')
my_data.sample(n=250).plot(kind='scatter', x='X Data', y='Y')
plt.plot(brand_new_data, predictions, 'r*')
%matplotlib
my_data.sample(n=250).plot(kind='scatter', x='X Data', y='Y')
plt.plot(brand_new_data, predictions, 'r*')

# More neural networking. Now with a readl CSV dataset
import tensorflow as tf
import pandas as pd
ls
!cd Tensorflow-Bootcamp-master/
!ls
ls
!ls Tensorflow-Bootcamp-master/
!ls Tensorflow-Bootcamp-master/
diabetes =pd.read_csv('./Tensorflow-Bootcamp-master/02-TensorFlow-Basics/pima-indians-diabetes.csv')
diabetes.head()
 diabetes.columns
#Normalize columns
cols_to_norm = ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps',
       'Insulin', 'BMI', 'Pedigree']
# Normalizing the columns with pandas
diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max()-x.min() ))
diabetes.head()
# Create the feature columns
import tensorflow as tf
diabetes.columns
num_preg = tf.feature_column.numeric_column('Number_pregnant')
plasma_gluc = tf.feature_column.numeric_column('Glucose_concentration')
dias_press = tf.feature_column.numeric_column('Blood_pressure')
tricep = tf.feature_column.numeric_column('Triceps')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
diabetes_pedigree = tf.feature_column.numeric_column('Pedigree')
age = tf.feature_column.numeric_column('Age')
# Those were the continuous values. Now for the non-continuous values. There are 2 ways for this: 1) vocabulary list and 2) hash bucket
# This works because you can see that Group has only 4 values
assigned_group = tf.feature_column.categorical_column_with_vocabulary_list('Group', ['A', 'B', 'C', 'D'])
# But what if a column has too many values to type out
# Hash bucket size will be the maximum amount of groups that you believe will be in that categorical column
assigned_group = tf.feature_column.categorical_column_with_hash_bucket('Group', hash_bucket_size=10)
# Those were the continuous values. Now for the non-continuous values. There are 2 ways for this: 1) vocabulary list and 2) hash bucket
# This works because you can see that Group has only 4 values
assigned_group = tf.feature_column.categorical_column_with_vocabulary_list('Group', ['A', 'B', 'C', 'D'])
# Age was a column we did not normalize. That is because we're going to assign it to a catagorical column. Sometimes you get more information this way. This is called Feature Engineering.
import matplotlib.pyplot as plt
%matplotlib
# Plot a simple histogram of the age column
diabetes['Age'].hist(bins=20)
# Most of the people are quite young. Then it goes down. Instead of treating this as a continous value, maybe you can bucket this together. Create some boundaries for each decade.
# The trick is using the age numeric column created earlier.
age_bucket = tf.feature_column.bucketized_column(age, boundaries=[20,30,40,50,60,70,80])
# Putting this all together
feat_cols = [num_preg, plasma_gluc, dias_press, tricep, insulin, bmi, diabetes_pedigree, assigned_group, age_bucket]
# TRAIN TEST SPLIT
x_data = diabetes.drop('Class', axis=1)
x_data.head()
labels = diabetes['Class']
# Labels
from sklearn.model_selection import train_test_split
train_test_split?
# Copy paste the code from the __doc__X_train, X_test, y_train, y_test = train_test_split(x_data, labels, test_size=0.3, random_state=101)
# Copy paste the code from the __doc__ 
X_train, X_test, y_train, y_test = train_test_split(
...     X, y, test_size=0.33, random_state=42
_train, X_test, y_train, y_test = train_test_split(x_da
          ...: ta, labels, test_size=0.3, random_state=101)
X_train, X_test, y_train, y_test = train_test_split(x_data, labels, test_size=0.3, random_state=101)
# Now create the model. First the input function
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=10, num_epochs=1000, shuffle=True)
# First a Linear Classifier. Number of classes is normally 2
model = tf.estimator.LinearClassifier(feature_column=feat_cols, n_classes=2)
# First a Linear Classifier. Number of classes is normally 2
model = tf.estimator.LinearClassifier(feature_columns=feat_cols, n_classes=2)
model.train(input_fn=input_func, steps=1000)
# Now evaluate the model
# Shuffle is false, because we want to make sure this is evaluated in the same order
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, y=y_test, batch=10, num_epochs=1, shuffle=False)
# Shuffle is false, because we want to make sure this is evaluated in the same order
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False)
results = model.evaluate(eval_input_func)
results
# the accuracy can be used for an ROC curve for binary classification
# Now to practice get predictions off of this
pred_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=10, num_epochs=1, shuffle=False)
predictions = model.predict(pred_input_func)
my_pred = list(predictions)
my_pred
clear
# Now a dense neural network classificer
dnn_model = tf.estimator.DNNClassifier(hidden_units=[10,10,10], feature_columns=feat_cols, n_classes=2)
# Unfortunately when you run this with the same input function, you get an error
dnn_model.train(input_fn = input_func, steps=1000)
# If you have a feature column and try to use this on a densely connected neural network, you have to pass it into an embedding_column. So the categorical columns are giving the trouble.
embedded_grouop_col = tf.feature_column.embedding_column(assigned_group, dimension=4)
%history -g feat_cols
# Replace the assigned_group with the embedded_group
feat_cols = [num_preg, plasma_gluc, dias_press, tricep, insulin, bmi, diabetes_pedigree, embedded_group, age_bucket]
# Replace the assigned_group with the embedded_group
feat_cols = [num_preg, plasma_gluc, dias_press, tricep, insulin, bmi, diabetes_pedigree, embedded_grouop_col, age_bucket]
input_func = tf.estimator.inputs.pandas_input_fn(X_train, y_train, batch_size=10, num_epochs=1000, shuffle=True)
dnn_model = tf.estimator.DNNClassifier(hidden_units=[10,10,10].feature_columns=feat_cols, n_classes=2)
# hidden_units are the number or neurons. The more neurons, the longer it takes
dnn_model = tf.estimator.DNNClassifier(hidden_units=[10,10,10],feature_columns=feat_cols, n_classes=2)
dnn_model.train(input_fn=input_func, steps=1000)
eval_input_func = tf.estimator.inputs.pandas_input_fun(x=X_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False)
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False)
dnn_model.evaluate(eval_input_func)
# We get allmost the same accuracy as before. So the DNN Network is performing the same.
%history
