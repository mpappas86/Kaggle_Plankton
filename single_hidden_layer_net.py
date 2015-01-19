from skimage.io import imread
from skimage.transform import resize
# from sklearn.ensemble import RandomForestClassifier as RF
# import glob
# import os
# from sklearn import cross_validation
# from sklearn.cross_validation import StratifiedKFold as KFold
# from sklearn.metrics import classification_report
# from matplotlib import pyplot as plt
# from matplotlib import colors
# from pylab import cm
# from skimage import segmentation, measure, morphology
# from skimage.morphology import watershed
# import numpy as np
# import pandas as pd
# from scipy import ndimage
# from skimage.feature import peak_local_max
# import shape_features as sf
# import grading
# import warnings
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

nnet = Neural_Net(num_inputs,num_outputs)
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

# Need to do xval instead of training on all data.  I'll do that tomorrow
t1 = nnet.train_mini(data, labels, mbsize=10, epochs=5, tag="5 epochs of training ", taginc=100)
