import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs


class Operation():
    def __init__(self, input_nodes=[]):
        self.input_nodes = input_nodes
        self.output_nodes = []

        for node in input_nodes:
            node.output_nodes.append(self)

        # Global variable to add everything to the graph
        _default_graph.operations.append(self)

    def compute(self):
        pass


# Tensorflow uses lowercase classes for Operation computations
class add(Operation):
    def __init__(self, x, y):
        super().__init__([x, y])  # Input nodes for operation

    # This compute will overwrite the Operation.compute()
    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var + y_var


class multiply(Operation):
    def __init__(self, x, y):
        super().__init__([x, y])

    # This compute will overwrite the Operation.compute()
    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var * y_var


class matmul(Operation):
    def __init__(self, x, y):
        super().__init__([x, y])

    # This compute will overwrite the Operation.compute()
    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        # This will be a numpy array, so this has a .dot method.
        return x_var.dot(y_var)


class Sigmoid(Operation):
    def __init__(self, z):
        super().__init__([z])

    def compute(self, z_val):
        return 1 / (1 + np.exp(-z_val))


class Placeholder():
    def __init__(self):
        self.output_nodes = []
        _default_graph.placeholders.append(self)


class Variable():
    def __init__(self, initial_value=None):
        self.value = initial_value
        self.output_nodes = []
        _default_graph.variables.append(self)


class Graph():
    def __init__(self):
        self.operations = []
        self.placeholders = []
        self.variables = []

    def set_as_default(self):
        global _default_graph
        _default_graph = self


def traverse_postorder(operation):
    """
    Postorder Traversal of Nodes. Makes sure computations are done in
    the correct order (Ax first, then Ax + b).
    """
    nodes_postorder = []

    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)

    recurse(operation)
    return nodes_postorder


class Session():

    def run(self, operation, feed_dict={}):
        nodes_postorder = traverse_postorder(operation)

        for node in nodes_postorder:
            if type(node) == Placeholder:
                node.output = feed_dict[node]

            elif type(node) == Variable:
                node.output = node.value

            else:
                # Operation
                node.inputs = [
                    input_node.output for input_node in node.input_nodes]
                node.output = node.compute(*node.inputs)

            if type(node.output) == list:
                node.output = np.array(node.output)

        return operation.output


'''
z = Ax + b
A = 10
b = 1
z = 10x + 1
'''

# Creating graph 1
g = Graph()
g.set_as_default()
A = Variable(10)
b = Variable(1)
x = Placeholder()
y = multiply(A, x)
z = add(y, b)

sess = Session()
result = sess.run(operation=z, feed_dict={x: 10})
print(result) # 101


# Graph 2
g = Graph()
g.set_as_default()
A = Variable([[10, 20],[30, 40]])
b = Variable([1, 2])
x = Placeholder()
y = matmul(A, x)
z = add(y, b)
sess = Session()
sess.run(Operation=z, feed_dict={x: 10})
#   [[101 202]
#    [301 402]]


# Sigmoid: Classification activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


sample_z = np.linspace(-10, 10, 100)
sample_a = sigmoid(sample_z)
plt.plot(sample_z, sample_a)
plt.show()


# Make a graph with random blobs
data = make_blobs(
    n_samples=50,
    n_features=2,
    centers=2,
    random_state=75
)
features = data[0]
labels = data[1]
plt.scatter(
    features[:, 0],
    features[:, 1],
    c=labels,
    cmap='coolwarm'
)
plt.show()

# Draw a manual line to seperate these blobs
# (draw a line through the middle)
x = np.linspace(0, 11, 10)
y = -x + 5
plt.scatter(
    features[:, 0],
    features[:, 1],
    c=labels,
    cmap='coolwarm'
)
plt.plot(x, y)
plt.show()


# Compute this line
arr1 = np.array([1, 1]).dot(np.array([[8], [10]])) - 5
arr2 = np.array([1, 1]).dot(np.array([[2], [-10]])) - 5

g = Graph()
g.set_as_default()
x = Placeholder()
w = Variable([1, 1])
b = Variable(-5)
z = add(matmul(w, x), b)
a = Sigmoid(z)
sess = Session()
sess.run(operation=a, feed_dict={x: [8, 10]})
# 0.99999
# Bijna 1, dus punt 8.10 op de grafiek behoort
# vrijwel zeker tot de rode bolletjes

sess = Session()
sess.run(operation=a, feed_dict={x: [2, -10]})
# 2.260 * 10^-6
# Bijna 0, dus punt 2,-10 behoort bijna zeker
# tot de blauwe bolletjes
