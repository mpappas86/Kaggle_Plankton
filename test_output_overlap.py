from neural_net import Neural_Net
from neural_node import Neural_Node
# from neural_layer import Neural_Layer
from sigmoid_layer import Sigmoid_Layer

nnet = Neural_Net(78,10)
input_layer1 = Sigmoid_Layer(60,30)
input_layer2 = Sigmoid_Layer(60,35)
hidden_layer1 = Sigmoid_Layer(30,40)
hidden_layer2 = Sigmoid_Layer(20,20)
# hidden node 3 will weight-share with hidden node 1
hidden_layer3 = Sigmoid_Layer(30,40)
output_layer1 = Sigmoid_Layer(55,5)
output_layer2 = Sigmoid_Layer(50,5)

input_node1 = Neural_Node(input_layer1)
input_node2 = Neural_Node(input_layer2)
hidden_node1 = Neural_Node(hidden_layer1)
hidden_node2 = Neural_Node(hidden_layer2)
hidden_node3 = Neural_Node(hidden_layer3)
output_node1 = Neural_Node(output_layer1)
output_node2 = Neural_Node(output_layer2)

# connect input and hidden nodes
# first 200 of hn1 go to first 200 in1
hidden_node1.add_input(input_node1, xrange(20))
input_node1.add_output(hidden_node1, xrange(20))
# last 100 of hn1 go to last 100 of in2
hidden_node1.add_input(input_node2, xrange(20,30))
input_node2.add_output(hidden_node1, xrange(25,35))
# first 150 of hn2 go to last 150 of in1
hidden_node2.add_input(input_node1, xrange(15))
input_node1.add_output(hidden_node2, xrange(15,30))
# last 50 of hn2 go to 200-250 of in2
hidden_node2.add_input(input_node2, xrange(15,20))
input_node2.add_output(hidden_node2, xrange(20,25))
# all of hn3 goes to 0-200 and 250-350 of in2
from itertools import chain
hidden_node3.add_input(input_node2, xrange(30))
input_node2.add_output(hidden_node3, list(chain(xrange(20), xrange(25,35))))

# connect hidden and output nodes
output_node1.add_input(hidden_node3, xrange(10))
output_node1.add_input(hidden_node2, xrange(10,30))
output_node1.add_input(hidden_node1, xrange(30,55))

output_node2.add_input(hidden_node3, xrange(30))
output_node2.add_input(hidden_node1, xrange(30,50))

hidden_node1.add_output(output_node2,xrange(20))
hidden_node1.add_output(output_node1,xrange(15,40))

hidden_node2.add_output(output_node1, xrange(20))

hidden_node3.add_output(output_node2, xrange(30))
hidden_node3.add_output(output_node1, xrange(30,40))

# add nodes to network
nnet.add_input(input_node1, (xrange(60),xrange(60)))
nnet.add_input(input_node2, (xrange(18,78),xrange(60)))
nnet.add_node(hidden_node1)
nnet.add_node(hidden_node2)
nnet.add_node(hidden_node3)
nnet.add_output(output_node1, (xrange(5),xrange(5)))
nnet.add_output(output_node2, (xrange(5,10),xrange(5)))
# nnet.add_output(hidden_node3, (xrange(9,10),xrange(399,400)))

# generate some data
data = np.random.rand(78,5)

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
#     costs.append(nnet.cost(data,labels))
    
# a = [x-y for x,y in zip(costs[0:(len(costs)-1)],costs[1:len(costs)])]
# print all([x > 0 for x in a])

input_layer1.set_lrates(0,0)
input_layer2.set_lrates(0,0)
hidden_layer1.set_lrates(0,0)
hidden_layer2.set_lrates(0,0)
hidden_layer3.set_lrates(0,0)
output_layer1.set_lrates(0,0)
output_layer2.set_lrates(0,0)

nnet.backprop(data, labels)
gradients = []
for node in nnet.nodes:
    gradients.append(np.concatenate((node.layer.Wgrad.flatten(), node.layer.bgrad.flatten())))
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
