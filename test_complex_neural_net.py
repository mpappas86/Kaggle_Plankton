from neural_net import Neural_Net
from neural_node import Neural_Node
# from neural_layer import Neural_Layer
from sigmoid_layer import Sigmoid_Layer

nnet = Neural_Net(784,10)
input_layer1 = Sigmoid_Layer(600,300)
input_layer2 = Sigmoid_Layer(600,350)
hidden_layer1 = Sigmoid_Layer(300,400)
hidden_layer2 = Sigmoid_Layer(200,200)
# hidden node 3 will weight-share with hidden node 1
# hidden_layer3 = Sigmoid_Layer(300,400)
output_layer1 = Sigmoid_Layer(550,5)
output_layer2 = Sigmoid_Layer(230,4)

input_node1 = Neural_Node(input_layer1)
input_node2 = Neural_Node(input_layer2)
hidden_node1 = Neural_Node(hidden_layer1)
hidden_node2 = Neural_Node(hidden_layer2)
hidden_node3 = Neural_Node(hidden_layer1)
output_node1 = Neural_Node(output_layer1)
output_node2 = Neural_Node(output_layer2)

# connect input and hidden nodes
# first 200 of hn1 go to first 200 in1
hidden_node1.add_input(input_node1, xrange(200))
input_node1.add_output(hidden_node1, xrange(200))
# last 100 of hn1 go to last 100 of in2
hidden_node1.add_input(input_node2, xrange(200,300))
input_node2.add_output(hidden_node1, xrange(250,350))
# first 150 of hn2 go to last 150 of in1
hidden_node2.add_input(input_node1, xrange(150))
input_node1.add_output(hidden_node2, xrange(150,300))
# last 50 of hn2 go to 200-250 of in2
hidden_node2.add_input(input_node2, xrange(150,200))
input_node2.add_output(hidden_node2, xrange(200,250))
# all of hn3 goes to 0-200 and 250-350 of in2
from itertools import chain
hidden_node3.add_input(input_node2, xrange(300))
input_node2.add_output(hidden_node3, list(chain(xrange(200), xrange(250,350))))

# connect hidden and output nodes
output_node1.add_input(hidden_node3, xrange(100))
output_node1.add_input(hidden_node2, xrange(100,300))
output_node1.add_input(hidden_node1, xrange(300,550))

output_node2.add_input(hidden_node3, xrange(130))
output_node2.add_input(hidden_node1, xrange(130,230))

hidden_node1.add_output(output_node2,xrange(100,200))
hidden_node1.add_output(output_node1,xrange(150,400))

hidden_node2.add_output(output_node1, xrange(200))

hidden_node3.add_output(output_node2, xrange(130))
hidden_node3.add_output(output_node1, xrange(300,400))

# add nodes to network
nnet.add_input(input_node1, (xrange(600),xrange(600)))
nnet.add_input(input_node2, (xrange(184,784),xrange(600)))
nnet.add_node(hidden_node1)
nnet.add_node(hidden_node2)
nnet.add_node(hidden_node3)
nnet.add_output(output_node1, (xrange(5),xrange(5)))
nnet.add_output(output_node2, (xrange(5,9),xrange(4)))
nnet.add_output(hidden_node3, (xrange(9,10),xrange(399,400)))

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
    costs.append(nnet.cost(data,labels))
    
a = [x-y for x,y in zip(costs[0:(len(costs)-1)],costs[1:len(costs)])]
print all([x > 0 for x in a])
