import numpy as np
np.random.seed(1)
import time

from DataReader.training_list import classcounts
from DataReader.data_selector import select_subset

import grading

from neural_net import Neural_Net
from neural_node import Neural_Node
from sigmoid_layer import Sigmoid_Layer
from softmax_layer import Softmax_Layer
from autoconnect import autoconnect


#This file should be where we make general modifications to our approach. EVERYTHING that might
#want to be general should be here - then in the particular Kaggle thing, we will have something
#that calls these methods with preset params.

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

def augment_with_features(data, feature_list, image_shape):
  if feature_list is None:
    return data
  else:
    image_height=image_shape[0]
    image_width=image_shape[1]
    image_size=image_width*image_height
    augmented_data = np.empty((image_size + len(feature_list), data.shape[1]))
    augmented_data[:image_size, :] = data
    reshaped_data = data.reshape(image_width, image_height, data.shape[1]))
    augmented_data[image_size:, :] = np.squeeze(np.array([feature_list[i](reshaped_data) for i in range(0, len(feature_list))]))
    return augmented_data

def deploy(rawdata, image_shape, nnet, num_epochs=10, feature_list=None):
  image_size = image_shape[0]*image_shape[1]

  print "Generating Validation Data"
  svaldata, sflags, iflags = select_subset(rawdata, 5, [0.7, 0.8])
  valdata = augment_with_features(svaldata[:image_size,:], feature_list, image_shape)
  vallabels = svaldata[image_size:,:]

  print "Generating Test Data"
  stestdata, sflags, iflags = select_subset(rawdata, 100, [0.8,1])
  testdata = augment_with_features(stestdata[:image_size,:], feature_list, image_shape)
  testlabels = stestdata[image_size:,:]

  ts = []
  vs = []
  predictions = []
  llos = []

  t0 = time.time()
  for epoch in xrange(num_epochs):
      print "Epoch "+str(epoch)
      seldata, sflags, iflags = select_subset(rawdata,480,[0,0.7])
      data = augment_with_features(seldata[:image_size,:], feature_list, image_shape)
      labels = seldata[image_size:,:]
      
      ttemp, vtemp = nnet.train_mini(data, labels, mbsize=100, epochs=1, tag="Epoch "+str(epoch)+" ", taginc=100, valid_data=valdata, valid_labels=vallabels)
      ts.append(ttemp)
      vs.append(vtemp)

      nnet.set_buffer_depth(testdata.shape[1])
      nnet.input_data(testdata)
      nnet.forward_pass()
      predictions.append(nnet.retrieve_output())

      llos.append(grading.multiclass_log_loss(testlabels.argmax(0), predictions[-1].T))
      print "Epoch "+str(epoch)+" log-loss:",llos[-1],"Time taken:",time.time()-t0
