# Variable nodes
# Store an initial value but can change
# Must call initializer to assign value

import tensorflow as tf

var_node_1 = tf.Variable([5.0], dtype=tf.float32)
const_node_1 = tf.constant([10.0], dtype=tf.float32)

session = tf.Session()
# To make this work you nee to initialize the variable node first. You need
# to call this once to initialize all global variables.
init = tf.global_variables_initializer()
session.run(init)

# res = session.run(var_node_1) : 5.0
print(session.run(var_node_1 * const_node_1))  # 50.0

# Reassign 10 to var_node_1
session.run(var_node_1.assign([10.0]))

print(session.run(var_node_1))
