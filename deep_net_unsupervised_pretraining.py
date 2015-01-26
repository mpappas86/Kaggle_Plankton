from skimage.io import imread
from skimage.transform import resize
import numpy as np
import os
import pickle
import time
np.random.seed(1)

from training_list import classcounts
import grading

from mllib.neural_net import Neural_Net
from mllib.neural_node import Neural_Node
from mllib.sigmoid_layer import Sigmoid_Layer
from mllib.softmax_layer import Softmax_Layer
from mllib.autoconnect import autoconnect

path = os.getcwd()

def rotate_images(imgs):
    ret = []
    for img in imgs:
        tmp = img
        for x in xrange(3):
            tmp = np.rot90(tmp)
            ret.append(tmp)
    return ret

def translate_images(imgs):
    ret = []
    for img in imgs:
        zs = [np.zeros(img.shape) for x in xrange(4)]
        zs[0][1:,1:] = img[:-1,:-1]
        zs[1][1:,:-1] = img[:-1,1:]
        zs[2][:-1,1:] = img[1:,:-1]
        zs[3][:-1,:-1] = img[1:,1:]
        ret.extend(zs)
    return ret

def flip_images(imgs):
    ret = []
    for img in imgs:
        ret.append(np.flipud(img))
    return ret

# Since this is non-deterministic, do this last
def add_noise(imgs, mean, sd):
    ret = []
    for img in imgs:
        ret.append(img+np.random.randn(img.shape[0],img.shape[1])*sd+mean)
    return ret

# 80x data multiplier
# time cost is roughly 1.5min / epoch
def transform_images(imgs):
    ret = []
    for img in imgs:
        tmp = []
        tmp.append(img)
        tmp.extend(flip_images(tmp))
        tmp.extend(rotate_images(tmp))
        tmp.extend(translate_images(tmp))
        tmp.extend(add_noise(tmp,0,0.01))
        ret.extend(tmp)
    return ret

labelnames = [x[0] for x in classcounts]
# Min = 9, Max = 1979

image_width = 25
image_height = 25
image_size = image_width*image_height

print "Loading Training Data"
# This guard lets me evaluate the whole file multiple times in one python run without reading in the data every time
try:
    len(rawdata)
except:
    with open('training_data_dictionary.npy','rb') as f:
        rawdata = np.load(f)[()]
        
def select_subset(alldata, desired_samples = 500, percent_range = [0,1]):
    ret = []
    samples_flag = [True]
    index_flag = [True]
    num_all_samples = 0
    for key, val in alldata.iteritems():
        lower_index = int(percent_range[0]*len(val))
        upper_index = int(percent_range[1]*len(val))
        if lower_index == upper_index:
            index_flag[0] = False
            index_flag.append(key)
            if upper_index == len(val):
                lower_index = lower_index - 1
            else:
                upper_index = upper_index + 1
        working_val = val[lower_index:upper_index]
        pool = np.array(transform_images(working_val))
        if pool.shape[0] < desired_samples:
            samples_flag[0] = False
            samples_flag.append(key)
            numsamples = pool.shape[0]
            choice = pool.reshape(numsamples, pool.shape[1]*pool.shape[2])
        else:
            numsamples = desired_samples
            choice = pool[np.random.choice(pool.shape[0], numsamples, replace=False),...].reshape(numsamples, pool.shape[1]*pool.shape[2])
        label = np.array([x == key for x in labelnames])
        rlabel = np.repeat(label[np.newaxis,:],numsamples,axis=0)
        labeledchoice = np.concatenate((choice, rlabel),axis=1)
        ret.append(labeledchoice)
        num_all_samples = num_all_samples + numsamples
    final = np.concatenate(ret)
    np.random.shuffle(final)
    return final.T, samples_flag, index_flag

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

# percent_range = [0.7, 0.8]
# tmp = []
# for key, val in rawdata.iteritems():
#     lower_index = int(percent_range[0]*len(val))
#     upper_index = int(percent_range[1]*len(val))
#     working_val = val[lower_index:upper_index]
#     npval = np.array(val).reshape(len(val),-1)
#     label = np.array([x == key for x in labelnames])
#     rlabel = np.repeat(label[np.newaxis,:],len(val),axis=0)
#     labeledchoice = np.concatenate((npval, rlabel),axis=1)
#     tmp.append(labeledchoice)
# svaldata = np.concatenate(tmp)
# np.random.shuffle(svaldata)
# svaldata = svaldata.T

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

glrate = 0.01
input_layer.set_lrates(glrate, glrate)
hidden_layer1.set_lrates(glrate, glrate)
hidden_layer2.set_lrates(glrate, glrate)
output_layer.set_lrates(glrate, glrate)

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
