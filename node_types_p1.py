import tensorflow as tf

# Constant nodes are a way to store a constant value within a
# particular node, used to perform operations like retrieve.
# Constant meaning unchangeable, like const
const_node_1 = tf.constant(1.0, dtype=tf.float32)
# Datatype can also be implied from value
const_node_2 = tf.constant(2.0, dtype=tf.float32)
# Const nodes can also be a tensor
const_node_3 = tf.constant([3.0, 4.0, 5.0], tf.float32)

# Operator/operation nodes
# These can change values 
adder_node_1 = tf.add(const_node_1, const_node_2)
adder_node_2 = const_node_1 + const_node_2
mult_node_1 = adder_node_2 * const_node_3

session = tf.Session()
# Creating an tensor (a type of array) of feeding the nodes
session.run([const_node_1, const_node_2])

print(session.run([const_node_1, const_node_2]))
print(session.run(adder_node_2))