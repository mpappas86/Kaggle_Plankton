from skimage.io import imread
from skimage.transform import resize
import os
np.random.seed(1)

from training_list import classcounts

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
    filepath = path + "/train/" + cls[0] + "/"
    for filename in os.walk(filepath).next()[2]:
        # print "reading in ", filepath+filename
        image = imread(filepath+filename, as_grey=True)
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
with open(r'deep_net_first_try.gv', 'w') as f:
    f.write(gfile)

train_index = int(data.shape[1]*0.7)
valid_index = int(data.shape[1]*0.9)
train_data = data[:,:train_index]
train_labels = labels[:,:train_index]
valid_data = data[:,train_index:valid_index]
valid_labels = labels[:,train_index:valid_index]
test_data = data[:,valid_index:]
test_labels = labels[:,valid_index:]

weights = nnet.get_weight_vector()
nnet.set_weight_vector(np.random.randn(weights.shape[0])*1e-10)

t1, v1 = nnet.train_mini(train_data, train_labels, mbsize=10, epochs=5, tag="5 epochs of uniform-prior training ", taginc=100, valid_data=valid_data, valid_labels=valid_labels)

nnet.set_buffer_depth(test_data.shape[1])
nnet.input_data(test_data)
nnet.forward_pass()
predicted_labels = nnet.retrieve_output()
import grading
llos = grading.multiclass_log_loss(test_labels.argmax(0), predicted_labels.T)

t2, v2 = nnet.train_mini(train_data, train_labels, mbsize=10, epochs=5, tag="5 more epochs of uniform-prior training ", taginc=100, valid_data=valid_data, valid_labels=valid_labels)

nnet.set_buffer_depth(test_data.shape[1])
nnet.input_data(test_data)
nnet.forward_pass()
predicted_labels2 = nnet.retrieve_output()
llos2 = grading.multiclass_log_loss(test_labels.argmax(0), predicted_labels2.T)

print "Cost after 5 epochs:", llos
print "Cost after 10 epochs:", llos2

t3, v3 = nnet.train_mini(train_data, train_labels, mbsize=10, epochs=50, tag="50 more epochs of uniform-prior training ", taginc=100, valid_data=valid_data, valid_labels=valid_labels)

nnet.set_buffer_depth(test_data.shape[1])
nnet.input_data(test_data)
nnet.forward_pass()
predicted_labels3 = nnet.retrieve_output()
llos3 = grading.multiclass_log_loss(test_labels.argmax(0), predicted_labels3.T)

print "Cost after 5 epochs:", llos
print "Cost after 10 epochs:", llos2
print "Cost after 60 epochs:", llos3

# import pickle
# with open(r'deep_net_first_try.pkl','w') as f:
#     pickle.dump(nnet.get_weight_vector(), f)
