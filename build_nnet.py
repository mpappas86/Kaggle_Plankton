import numpy as np
np.random.seed(1)

from DataReader.training_list import classcounts

from mllib.neural_net import Neural_Net
from mllib.neural_node import Neural_Node
from mllib.sigmoid_layer import Sigmoid_Layer
from mllib.softmax_layer import Softmax_Layer
from mllib.autoconnect import autoconnect

def build(image_size, glrate):
  num_inputs = image_size
  num_hiddens = num_inputs
  num_outputs = len(classcounts)

  cf = lambda x,y: -(y*np.log(np.maximum(np.minimum(x,1-(1e-15)),1e-15))).sum(0)
  dcf = lambda x,y: -y/np.maximum(np.minimum(x,1-(1e-15)),1e-15)

  nnet = Neural_Net(num_inputs,num_outputs,cost_function=cf,dcost=dcf)
  input_layer = Sigmoid_Layer(num_inputs,num_hiddens)
  hidden_layer1 = Sigmoid_Layer(num_hiddens, num_hiddens)
  hidden_layer2 = Sigmoid_Layer(num_hiddens, num_hiddens)
  output_layer = Softmax_Layer(num_hiddens,num_outputs,1)

  input_node = Neural_Node(input_layer)
  hidden_node1 = Neural_Node(hidden_layer1)
  hidden_node2 = Neural_Node(hidden_layer2)
  output_node = Neural_Node(output_layer)

  connections = [("input", input_node, num_inputs),
                 (input_node, hidden_node1, num_hiddens),
                 (hidden_node1, hidden_node2, num_hiddens),
                 (hidden_node2, output_node, num_hiddens),
                 (output_node, "output", num_outputs)]

  gfile = autoconnect(nnet, connections)
  # with open(r'deep_net_first_try.gv', 'w') as f:
  #     f.write(gfile)

  weights = nnet.get_weight_vector()
  nnet.set_weight_vector(np.random.randn(weights.shape[0])*1e-10)

  input_layer.set_lrates(glrate, glrate)
  hidden_layer1.set_lrates(glrate, glrate)
  hidden_layer2.set_lrates(glrate, glrate)
  output_layer.set_lrates(glrate, glrate)

  return nnet, gfile