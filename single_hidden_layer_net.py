from skimage.io import imread
from skimage.transform import resize
import os
import numpy as np
np.random.seed(1)

from DataReader.training_list import classcounts

from mllib.neural_net import Neural_Net
from mllib.neural_node import Neural_Node
from mllib.sigmoid_layer import Sigmoid_Layer
from mllib.softmax_layer import Softmax_Layer
from mllib.autoconnect import autoconnect

path = os.getcwd()

labelnames = [x[0] for x in classcounts]

image_width = 25
image_height = 25
image_size = image_width*image_height

# TODO: I'm just going to lump everything into one data vector, without worrying about evenly sampling classes.  This is an area of improvement 30336
rawdata = []
counter = 0
for cls in classcounts:
    counter = counter + 1
    print "Importing " + cls[0], str(counter) + " of " + str(len(classcounts))
    label = np.array([x == cls[0] for x in labelnames])
    filepath = os.path.join(path, "train", cls[0])
    for filename in os.walk(filepath).next()[2]:
        if filename[-4:] != ".jpg":
              continue
        # print "reading in ", filepath+filename
        image = imread(os.path.join(filepath,filename), as_grey=True)
        rimage = resize(image, (image_height, image_width), order=1, mode="constant", cval=0.0)
        rawdata.append(np.concatenate((rimage.reshape((image_size)),label)))
rawdata = np.array(rawdata)
np.random.shuffle(rawdata)
rawdata = rawdata.T
data = rawdata[:image_size,:]
labels = rawdata[image_size:,:]

num_inputs = image_size
num_hiddens = num_inputs
num_outputs = len(classcounts)

cf = lambda x,y: -(y*np.log(np.maximum(np.minimum(x,1-(1e-15)),1e-15))).sum(0)
dcf = lambda x,y: -y/np.maximum(np.minimum(x,1-(1e-15)),1e-15)

nnet = Neural_Net(num_inputs,num_outputs,cost_function=cf,dcost=dcf)
input_layer = Sigmoid_Layer(num_inputs,num_hiddens)
output_layer = Softmax_Layer(num_hiddens,num_outputs,1)

input_node = Neural_Node(input_layer)
output_node = Neural_Node(output_layer)

connections = [("input", input_node, num_inputs),
               (input_node, output_node, num_hiddens),
               (output_node, "output", num_outputs)]

gfile = autoconnect(nnet, connections)
with open(r'single_hidden_layer_net.gv', 'w') as f:
    f.write(gfile)

train_index = int(data.shape[1]*0.7)
valid_index = int(data.shape[1]*0.9)
train_data = data[:,:train_index]
train_labels = labels[:,:train_index]
valid_data = data[:,train_index:valid_index]
valid_labels = labels[:,train_index:valid_index]
test_data = data[:,valid_index:]
test_labels = labels[:,valid_index:]

t1, v1 = nnet.train_mini(train_data, train_labels, mbsize=10, epochs=10, tag="5 epochs of random-prior training ", taginc=100, valid_data=valid_data, valid_labels=valid_labels)

nnet.set_buffer_depth(test_data.shape[1])
nnet.input_data(test_data)
nnet.forward_pass()
random_predicted_labels = nnet.retrieve_output()
import grading
random_llos = grading.multiclass_log_loss(test_labels.argmax(0), random_predicted_labels.T)

weights = nnet.get_weight_vector()
nnet.set_weight_vector(np.random.randn(weights.shape[0])*1e-10)

t1, v1 = nnet.train_mini(train_data, train_labels, mbsize=10, epochs=10, tag="5 epochs of uniform-prior training ", taginc=100, valid_data=valid_data, valid_labels=valid_labels)

nnet.set_buffer_depth(test_data.shape[1])
nnet.input_data(test_data)
nnet.forward_pass()
prior_predicted_labels = nnet.retrieve_output()
import grading
prior_llos = grading.multiclass_log_loss(test_labels.argmax(0), prior_predicted_labels.T)

print "Random prior cost after 5 epochs:", random_llos
print "Uniform prior cost after 5 epochs:", prior_llos

# import pickle
# with open(r'single_hidden_layer_net.pkl','w') as f:
#     pickle.dump(weights, f)
