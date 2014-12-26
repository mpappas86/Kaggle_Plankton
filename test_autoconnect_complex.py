from neural_net import Neural_Net
from neural_node import Neural_Node
# from neural_layer import Neural_Layer
from sigmoid_layer import Sigmoid_Layer
from maxpool_layer import Maxpool_Layer
from autoconnect import autoconnect

nnet = Neural_Net(101,10)
input_layer1 = Sigmoid_Layer(25,15)
input_layer2 = Sigmoid_Layer(51,14)
hidden_layer = Maxpool_Layer(2)
output_layer1 = Sigmoid_Layer(10,6)
output_layer2 = Sigmoid_Layer(12,4)

input_node1 = Neural_Node(input_layer1, name="Input 1")
input_node2 = Neural_Node(input_layer1, name="Input 2")
input_node3 = Neural_Node(input_layer2, name="Input 3")
hidden_nodes = []
for x in xrange(22):
    hidden_nodes.append(Neural_Node(hidden_layer, name="Maxpool "+str(x)))
output_node1 = Neural_Node(output_layer1, name="Output 1")
output_node2 = Neural_Node(output_layer2, name="Output 2")

# Connect the inputs to the input nodes
connections = [("input", input_node1, 25),
               ("input", input_node2, 25),
               ("input", input_node3, 51)]

# Connect the first 7 maxpool nodes to input_node1
for x in xrange(7):
    connections.append((input_node1, hidden_nodes[x], 2))
# Connect the 8th maxpool node to input_node1 and input_node2
connections.append((input_node1, hidden_nodes[7], 1))
connections.append((input_node2, hidden_nodes[7], 1))
# Connect the 9th through 15th maxpool nodes to input_node2
for x in xrange(8,15):
    connections.append((input_node2, hidden_nodes[x], 2))
# Connect the 16th through 22nd maxpool nodes to input_node3
for x in xrange(15,22):
    connections.append((input_node3, hidden_nodes[x], 2))

# Connect the first 10 maxpool nodes to output_node1
for x in xrange(10):
    connections.append((hidden_nodes[x], output_node1, 1))
# Connect the second 12 maxpool nodes to output_node2
for x in xrange(10,22):
    connections.append((hidden_nodes[x], output_node2, 1))

# Connect the output nodes to the outputs
connections.extend([(output_node1, "output", 6),
                 (output_node2, "output", 4)])
    
autoconnect(nnet, connections)

# generate some data
data = np.random.rand(input_layer.W.shape[1],5)

# test a forward pass
nnet.set_buffer_depth(data.shape[1])
nnet.input_data(data)
nnet.forward_pass()
out_data = nnet.retrieve_output()

# generate labels for the data
labels = (np.random.rand(output_layer.shape[0],data.shape[1]) > 0.5).astype(float)

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
input_layer1.set_lrates(0,0)
input_layer2.set_lrates(0,0)
hidden_layer.set_lrates(0,0)
output_layer1.set_lrates(0,0)
output_layer2.set_lrates(0,0)

nnet.backprop(data, labels)
gradients = []
layerset = nnet.get_layerset()
for layer in layerset:
    try:
        gradients.append(np.concatenate((layer.Wgrad.flatten(), layer.bgrad.flatten())))
    except:
        pass
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
