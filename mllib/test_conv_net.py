from neural_net import Neural_Net
from neural_node import Neural_Node
# from neural_layer import Neural_Layer
from sigmoid_layer import Sigmoid_Layer
from maxpool_layer import Maxpool_Layer

nnet = Neural_Net(784,10)
# Currently at 28x28 input
input_layer1 = Sigmoid_Layer(9,1)
input_layer2 = Sigmoid_Layer(9,1)
input_layer3 = Sigmoid_Layer(9,1)
input_layer4 = Sigmoid_Layer(9,1)
# Now at 26x26x4
maxpool_layer = Maxpool_Layer(9)
# Now at 9x9x4
hidden_layer11 = Sigmoid_Layer(9*4,1)
hidden_layer12 = Sigmoid_Layer(9*4,1)
hidden_layer13 = Sigmoid_Layer(9*4,1)
hidden_layer14 = Sigmoid_Layer(9*4,1)
hidden_layer15 = Sigmoid_Layer(9*4,1)
hidden_layer16 = Sigmoid_Layer(9*4,1)
# Now at 7x7x6
# maxpool_layer2 = Maxpool_Layer(9)
# Now at 3x3x6 = 54
hidden_layer2 = Sigmoid_Layer(54,28)
# Now at 28
output_layer = Sigmoid_Layer(28,10)
# Finally at 10 output

input_node1s = []
input_node2s = []
input_node3s = []
input_node4s = []
inputs = [input_node1s, input_node2s, input_node3s, input_node4s]
maxpool_node11s = []
maxpool_node12s = []
maxpool_node13s = []
maxpool_node14s = []
maxpool1s = [maxpool_node11s, maxpool_node12s, maxpool_node13s, maxpool_node14s]
hidden_node11s = []
hidden_node12s = []
hidden_node13s = []
hidden_node14s = []
hidden_node15s = []
hidden_node16s = []
hiddens = [hidden_node11s, hidden_node12s, hidden_node13s, hidden_node14s, hidden_node15s, hidden_node16s]
maxpool_node21s = []
maxpool_node22s = []
maxpool_node23s = []
maxpool_node24s = []
maxpool_node25s = []
maxpool_node26s = []
maxpool2s = [maxpool_node21s, maxpool_node22s, maxpool_node23s, maxpool_node24s, maxpool_node25s, maxpool_node26s]

for x in xrange(26*26):
    input_node1s.append(Neural_Node(input_layer1, name="Input1_"+str(x)))
    input_node2s.append(Neural_Node(input_layer2, name="Input2_"+str(x)))
    input_node3s.append(Neural_Node(input_layer3, name="Input3_"+str(x)))
    input_node4s.append(Neural_Node(input_layer4, name="Input4_"+str(x)))
for x in xrange(9*9):
    maxpool_node11s.append(Neural_Node(maxpool_layer, name="Maxpool1.1_"+str(x)))
    maxpool_node12s.append(Neural_Node(maxpool_layer, name="Maxpool1.2_"+str(x)))
    maxpool_node13s.append(Neural_Node(maxpool_layer, name="Maxpool1.3_"+str(x)))
    maxpool_node14s.append(Neural_Node(maxpool_layer, name="Maxpool1.4_"+str(x)))
for x in xrange(7*7):
    hidden_node11s.append(Neural_Node(hidden_layer11, name="Hidden1.1_"+str(x)))
    hidden_node12s.append(Neural_Node(hidden_layer12, name="Hidden1.2_"+str(x)))
    hidden_node13s.append(Neural_Node(hidden_layer13, name="Hidden1.3_"+str(x)))
    hidden_node14s.append(Neural_Node(hidden_layer14, name="Hidden1.4_"+str(x)))
    hidden_node15s.append(Neural_Node(hidden_layer15, name="Hidden1.5_"+str(x)))
    hidden_node16s.append(Neural_Node(hidden_layer16, name="Hidden1.6_"+str(x)))
for x in xrange(3*3):
    maxpool_node21s.append(Neural_Node(maxpool_layer, name="Maxpool2.1_"+str(x)))
    maxpool_node22s.append(Neural_Node(maxpool_layer, name="Maxpool2.2_"+str(x)))
    maxpool_node23s.append(Neural_Node(maxpool_layer, name="Maxpool2.3_"+str(x)))
    maxpool_node24s.append(Neural_Node(maxpool_layer, name="Maxpool2.4_"+str(x)))
    maxpool_node25s.append(Neural_Node(maxpool_layer, name="Maxpool2.5_"+str(x)))
    maxpool_node26s.append(Neural_Node(maxpool_layer, name="Maxpool2.6_"+str(x)))

hidden_node2 = Neural_Node(hidden_layer2, name="Hidden2")
output_node = Neural_Node(output_layer, name="Output")

# Do the manual filter connections first
# Select which filter we're using
for hd in xrange(4):
    # Select the appropriate node and pixel subsets
    for row in xrange(26):
        for col in xrange(26):
            base = row*26+col
            base2 = (row+1)*26+col
            base3 = (row+2)*26+col
            nnet.add_input(inputs[hd][base], ([base, base+1, base+2, base2, base2+1, base2+2, base3, base3+1, base3+2], xrange(9)))

# Select which filter we're using
for hd in xrange(6):
    # Select the appropriate node
    for hdrow in xrange(7):
        for hdcol in xrange(7):
            node = hiddens[hd][hdrow*7+hdcol]
            # Select each appropriate maxpool node
            for md in xrange(4):
                for mdrow in xrange(hdrow,hdrow+3):
                    for mdcol in xrange(hdcol,hdcol+3):
                        mp = maxpool1s[md][mdrow*9+mdcol]
                        # Attach the filter node to the maxpool node
                        node.add_input(mp, [md*9+(mdrow-hdrow)*3+(mdcol-hdcol)])
                        mp.add_output(node, [0])
                        
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
