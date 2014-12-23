from neural_net import Neural_Net
from neural_node import Neural_Node
# from neural_layer import Neural_Layer
from sigmoid_layer import Sigmoid_Layer

nnet = Neural_Net(784,10)
input_layer = Sigmoid_Layer(784,784)
hidden_layer = Sigmoid_Layer(784,784)
output_layer = Sigmoid_Layer(784,10)

input_node = Neural_Node(input_layer)
hidden_node = Neural_Node(hidden_layer)
output_node = Neural_Node(output_layer)

# connect input and hidden nodes
# connect full input range of hidden_node to full output range of output_node
hidden_node.add_input(input_node, xrange(784))
# connect full output range of input_node to full input range of hidden_node
input_node.add_output(hidden_node, xrange(784))

# connect hidden and output nodes
# connect full input range of output_node to full output range of hidden_node
output_node.add_input(hidden_node, xrange(784))
# connect full output range of hidden_node to full input range of output_node
hidden_node.add_output(output_node, xrange(784))

# add nodes to network
nnet.add_input(input_node, xrange(784))
nnet.add_node(hidden_node)
nnet.add_output(output_node, xrange(10))

# generate some data
data = np.random.rand(784,100)

# test a forward pass
nnet.set_buffer_depth(data.shape[1])
nnet.input_data(data)
nnet.forward_pass()
out_data = nnet.retrieve_output()

# generate labels for the data
labels = (np.random.rand(10,data.shape[1]) > 0.5).astype(float)

# test backpropogation
print nnet.cost(data, labels)
costs = [nnet.cost(data, labels)]
for x in xrange(100):
    nnet.backprop(data, labels)
    print nnet.cost(data, labels)
    costs.append(nnet.cost(data, labels))

a = [x-y for x,y in zip(costs[0:(len(costs)-1)],costs[1:len(costs)])]
print all([x > 0 for x in a])
