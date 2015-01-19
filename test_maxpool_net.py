from neural_net import Neural_Net
from neural_node import Neural_Node
# from neural_layer import Neural_Layer
from sigmoid_layer import Sigmoid_Layer
from maxpool_layer import Maxpool_Layer

nnet = Neural_Net(101,10)
input_layer = Sigmoid_Layer(101,44)
hidden_layer = Maxpool_Layer(2)
output_layer = Sigmoid_Layer(22,10)

input_node = Neural_Node(input_layer, name="Input")
hidden_nodes = []
for x in xrange(22):
    hidden_nodes.append(Neural_Node(hidden_layer, name="Maxpool "+str(x)))
output_node = Neural_Node(output_layer, name="Output")

# connect input and hidden nodes
for x in xrange(22):
    hidden_nodes[x].add_input(input_node, xrange(2))

for x in xrange(22):
    input_node.add_output(hidden_nodes[x], xrange(2*x,2*x+2))

# connect hidden and output nodes
for x in xrange(22):
    output_node.add_input(hidden_nodes[x], xrange(x,x+1))

for x in xrange(22):
    hidden_nodes[x].add_output(output_node, xrange(1))

# add nodes to network
nnet.add_input(input_node, (xrange(input_layer.W.shape[1]), xrange(input_layer.W.shape[1])))
for node in hidden_nodes:
    nnet.add_node(node)
nnet.add_output(output_node, (xrange(output_layer.W.shape[0]), xrange(output_layer.W.shape[0])))

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
input_layer.set_lrates(0,0)
hidden_layer.set_lrates(0,0)
output_layer.set_lrates(0,0)

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
