from skimage.io import imread
from skimage.transform import resize
import numpy as np
import os
import pickle
import time
np.random.seed(1)

from DataReader.training_list import classcounts
import grading

from mllib.neural_net import Neural_Net

from DataReader.data_selector import select_subset
from DataReader.read_images import read_training

import build_nnet

path = os.getcwd()

labelnames = [x[0] for x in classcounts]
# Min = 9, Max = 1979

image_width = 25
image_height = 25
image_size = image_width*image_height

print "Loading Training Data"
# This guard lets me evaluate the whole file multiple times in one python run without reading in the data every time
rawdata = read_training(image_width, image_height)

nnet, gfile = build_nnet.build(image_size, glrate=0.01)

print "Generating Validation Data"
svaldata, sflags, iflags = select_subset(rawdata, 5, [0.7, 0.8])
valdata = svaldata[:image_size,:]
vallabels = svaldata[image_size:,:]

print "Generating Test Data"
stestdata, sflags, iflags = select_subset(rawdata, 100, [0.8,1])
testdata = stestdata[:image_size,:]
testlabels = stestdata[image_size:,:]

ts = []
vs = []
predictions = []
llos = []

t0 = time.time()
for epoch in xrange(10):
    print "Epoch "+str(epoch)
    seldata, sflags, iflags = select_subset(rawdata,480,[0,0.7])
    data = seldata[:image_size,:]
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
