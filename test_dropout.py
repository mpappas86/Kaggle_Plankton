from neural_net import Neural_Net
from neural_node import Neural_Node
from sigmoid_layer import Sigmoid_Layer
from softmax_layer import Softmax_Layer
from autoconnect import autoconnect

import numpy as np

nnet = Neural_Net(101, 10, p=0.8)
input_layer = Sigmoid_Layer(101,37)
hidden_layer = Sigmoid_Layer(37,22,2)
output_layer = Softmax_Layer(22,10,1)

input_node = Neural_Node(input_layer, name="Input", p=0.7)
hidden_node = Neural_Node(hidden_layer, name="Hidden", p=0.9)
output_node = Neural_Node(output_layer, name="Output", p=None)

connections = [("input", input_node, 101),
               (input_node, hidden_node, 37),
               (hidden_node, output_node, 22),
               (output_node, "output", 10)]

gfile = autoconnect(nnet, connections)
with open(r'autoconnect_simple.gv', 'w') as f:
    f.write(gfile)

# generate some data
data = np.random.rand(input_layer.W.shape[1],5)

# test a forward pass
nnet.set_buffer_depth(data.shape[1])
nnet.input_data(data)
nnet.forward_pass()
out_data = nnet.retrieve_output()

# generate labels for the data
labels = (np.random.rand(10,data.shape[1]) > 0.5).astype(float)

# test backpropogation
# print nnet.cost(data, labels)
costs = [nnet.cost(data, labels)]
for x in xrange(100):
    nnet.backprop(data, labels)
    # print nnet.cost(data, labels)
    costs.append(nnet.cost(data, labels))

backprop_diffs = [x-y for x,y in zip(costs[0:(len(costs)-1)],costs[1:len(costs)])]
backprop_test = all([x > 0 for x in backprop_diffs])
# print all([x > 0 for x in a])

# gradient check
input_layer.set_lrates(0,0)
hidden_layer.set_lrates(0,0)
output_layer.set_lrates(0,0)

nnet.backprop(data, labels)
gradients = []
layerset = nnet.get_ordered_layerset()
for layer in layerset:
    gradients.append(np.concatenate((layer.Wgrad.flatten(), layer.bgrad.flatten())))
grads = np.concatenate(gradients)

weights = nnet.get_weight_vector()
epsilon = 1e-6
gradient_test = []
import copy
for index in xrange(len(weights)):
    new_weights = copy.copy(weights)
    new_weights[index] = weights[index] + epsilon
    nnet.set_weight_vector(new_weights)
    cost_up = nnet.cost(data, labels)
    new_weights[index] = weights[index] - epsilon
    nnet.set_weight_vector(new_weights)
    cost_down = nnet.cost(data, labels)
    gradient_test.append((cost_up - cost_down)/(2.0*epsilon))
    # if index % 100 == 0:
    #     print index, len(weights)
grad_test = np.array(gradient_test)

errors = np.abs(grads - grad_test)

grad_test_mean = np.mean(errors)
grad_test_max = np.max(errors)
