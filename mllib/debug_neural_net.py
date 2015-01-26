from neural_net import Neural_Net
from neural_node import Neural_Node
# from neural_layer import Neural_Layer
from sigmoid_layer import Sigmoid_Layer

nnet = Neural_Net(11,5)
input_layer = Sigmoid_Layer(11,9)
input_layer.set_lrates(0,0)
hidden_layer = Sigmoid_Layer(9,6)
hidden_layer.set_lrates(0,0)
output_layer = Sigmoid_Layer(6,5)
output_layer.set_lrates(0,0)

input_node = Neural_Node(input_layer,name="input")
hidden_node = Neural_Node(hidden_layer,name="hidden")
output_node = Neural_Node(output_layer,name="output")

# connect input and hidden nodes
# connect full input range of hidden_node to full output range of output_node
hidden_node.add_input(input_node, xrange(9))
# connect full output range of input_node to full input range of hidden_node
input_node.add_output(hidden_node, xrange(9))

# connect hidden and output nodes
# connect full input range of output_node to full output range of hidden_node
output_node.add_input(hidden_node, xrange(6))
# connect full output range of hidden_node to full input range of output_node
hidden_node.add_output(output_node, xrange(6))

# add nodes to network
nnet.add_input(input_node, xrange(11))
nnet.add_node(hidden_node)
nnet.add_output(output_node, xrange(5))

# generate some data
data = np.random.rand(11,1)

import numpy as np
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# test a forward pass
nnet.set_buffer_depth(data.shape[1])
nnet.input_data(data)
nnet.forward_pass()
out_data = nnet.retrieve_output()

inout = sigmoid(input_layer.W.dot(data)+input_layer.b)
print ((input_node.output_buffer - inout)**2).sum()
hidout = sigmoid(hidden_layer.W.dot(inout)+hidden_layer.b)
print ((hidden_node.output_buffer-hidout)**2).sum()
outout = sigmoid(output_layer.W.dot(hidout)+output_layer.b)
print ((output_node.output_buffer-outout)**2).sum()
print ((nnet.retrieve_output()-outout)**2).sum()

# generate labels for the data
labels = (np.random.rand(5,data.shape[1]) > 0.5).astype(float)

# test a training forward pass
nnet.input_data(data)
# nnet.forward_pass("training_push_output()")
predicted = nnet.retrieve_output()
print ((predicted-outout)**2).sum()

# test backpropogation
lr = 0.01

# print nnet.cost(data,labels)
# for x in xrange(10):
#     nnet.backprop(data, labels)
#     print nnet.cost(data,labels)
nnet.backprop(data,labels)
delta_last = (predicted-labels)*predicted*(1-predicted)
print
print (((predicted-labels) - output_node.output_buffer)**2).sum()
delta_hid = output_layer.W.T.dot(delta_last)*hidout*(1-hidout)
print ((output_layer.W.T.dot(delta_last)-hidden_node.output_buffer)**2).sum()
delta_in = hidden_layer.W.T.dot(delta_hid)*inout*(1-inout)
print ((hidden_layer.W.T.dot(delta_hid)-input_node.output_buffer)**2).sum()
print ((input_layer.W.T.dot(delta_in)-input_node.input_buffer)**2).sum()
