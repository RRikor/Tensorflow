# Placeholder nodes
# Nodes with no current value
# Pass in value when running session

# Explanation about shapes
# 5.0 shape = (,) | This has a shape of nothing
# [5.0] shape = (1,) There is 1 member in this array, but there are no
#   other tensors to take on additional shapes.
# [[5.0]] shape = (1,1) There is 1 array in the outer array and 1 member
#   within the array.
# [[5.0,4.0], [1,0,2.0]] shape = (2,2) There are 2 inner arrays each with
#   2 members.

import tensorflow as tf

# Since this is a placeholder, it does not need a value straight
# away.
placeholder_1 = tf.placeholder(dtype=tf.float32)
placeholder_2 = tf.placeholder(dtype=tf.float32)

# This assigns to placeholder_1 the values 1.0, 2.0, 3.0 and will then run
# placeholder_1 with the values.
# print(session.run(placeholder_1, {placeholder_1: [1.0, 2.0, 3.0]}))

multiply_node_1 = placeholder_1 * 3
multiply_node_2 = placeholder_1 * placeholder_2

# 4*2 = 8 | 5*4 = 20 | output --> [8. 20.]
session = tf.Session()
print(session.run(multiply_node_2, {
    placeholder_1: 4.0,
    placeholder_2: [2.0, 5.0]
    }))
