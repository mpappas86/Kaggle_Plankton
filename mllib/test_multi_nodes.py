from neural_net import Neural_Net
from neural_node import Neural_Node
# from neural_layer import Neural_Layer
from sigmoid_layer import Sigmoid_Layer

nnet = Neural_Net(101,10)
input_layer = Sigmoid_Layer(101,38)
hidden_layer = Sigmoid_Layer(19,11)
hidden_layer2 = Sigmoid_Layer(19,11)
output_layer = Sigmoid_Layer(22,10)

input_node = Neural_Node(input_layer)
hidden_node = Neural_Node(hidden_layer)
hidden_node2 = Neural_Node(hidden_layer2)
output_node = Neural_Node(output_layer)

# connect input and hidden nodes
hidden_node.add_input(input_node, xrange(19))
hidden_node2.add_input(input_node, xrange(19))

input_node.add_output(hidden_node, xrange(19))
input_node.add_output(hidden_node2, xrange(19,38))

# connect hidden and output nodes
output_node.add_input(hidden_node, xrange(11))
output_node.add_input(hidden_node2, xrange(11,22))

hidden_node.add_output(output_node, xrange(11))
hidden_node2.add_output(output_node, xrange(11))

# add nodes to network
nnet.add_input(input_node, (xrange(input_layer.W.shape[1]), xrange(input_layer.W.shape[1])))
nnet.add_node(hidden_node)
nnet.add_node(hidden_node2)
nnet.add_output(output_node, (xrange(output_layer.W.shape[0]), xrange(output_layer.W.shape[0])))

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
# costs = [nnet.cost(data, labels)]
# for x in xrange(100):
#     nnet.backprop(data, labels)
#     print nnet.cost(data, labels)
#     costs.append(nnet.cost(data, labels))

# a = [x-y for x,y in zip(costs[0:(len(costs)-1)],costs[1:len(costs)])]
# print all([x > 0 for x in a])

# gradient check
input_layer.set_lrates(0,0)
hidden_layer.set_lrates(0,0)
hidden_layer2.set_lrates(0,0)
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
    if index % 100 == 0:
        print index, len(weights)
grad_test = np.array(gradient_test)

errors = np.abs(grads - grad_test)
print np.mean(errors)
print np.max(errors)
