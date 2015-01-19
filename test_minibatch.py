from neural_net import Neural_Net
from neural_node import Neural_Node
from sigmoid_layer import Sigmoid_Layer

nnet = Neural_Net(101,10)
input_layer = Sigmoid_Layer(101,37)
hidden_layer = Sigmoid_Layer(37,22)
output_layer = Sigmoid_Layer(22,10)

input_node = Neural_Node(input_layer, name="Input")
hidden_node = Neural_Node(hidden_layer)
output_node = Neural_Node(output_layer)

# connect input and hidden nodes
# connect full input range of hidden_node to full output range of output_node
hidden_node.add_input(input_node, xrange(input_layer.W.shape[0]))
# connect full output range of input_node to full input range of hidden_node
input_node.add_output(hidden_node, xrange(hidden_layer.W.shape[1]))

# connect hidden and output nodes
# connect full input range of output_node to full output range of hidden_node
output_node.add_input(hidden_node, xrange(hidden_layer.W.shape[0]))
# connect full output range of hidden_node to full input range of output_node
hidden_node.add_output(output_node, xrange(output_layer.W.shape[1]))

# add nodes to network
nnet.add_input(input_node, (xrange(input_layer.W.shape[1]), xrange(input_layer.W.shape[1])))
nnet.add_node(hidden_node)
nnet.add_output(output_node, (xrange(output_layer.W.shape[0]), xrange(output_layer.W.shape[0])))

# generate some data
data = np.random.rand(input_layer.W.shape[1],10000)

# test a forward pass
nnet.set_buffer_depth(data.shape[1])
nnet.input_data(data)
nnet.forward_pass()
out_data = nnet.retrieve_output()

# generate labels for the data
labels = (np.random.rand(10,data.shape[1]) > 0.5).astype(float)

# test minibatch gradient descent
tst = nnet.train_mini(data, labels, mbsize=10, epochs=5, tag='test ', taginc=100)
