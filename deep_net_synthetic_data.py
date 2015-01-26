from skimage.io import imread
from skimage.transform import resize
import numpy as np
import os
import pickle
import time
np.random.seed(1)

from training_list import classcounts

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
        tmp.extend(add_noise(tmp,0,0.1))
        ret.extend(tmp)
    return ret

labelnames = [x[0] for x in classcounts]
# Min = 9, Max = 1979

image_width = 25
image_height = 25
image_size = image_width*image_height

try:
    len(rawdata)
except:
    with open('training_data_dictionary.pkl','rb') as f:
        rawdata = pickle.load(f)
    # rawdata = {}
    # counter = 0
    # for cls in classcounts:
    #     counter = counter + 1
    #     print "Importing " + cls[0], str(counter) + " of " + str(len(classcounts))
    #     # label = np.array([x == cls[0] for x in labelnames])
    #     filepath = path + "/train/" + cls[0] + "/"
    #     images = []
    #     for filename in os.walk(filepath).next()[2]:
    #         image = imread(filepath+filename, as_grey=True)
    #         rimage = resize(image, (image_height, image_width), order=1, mode="constant", cval=0.0)
    #         # images.append(np.concatenate((rimage.reshape((image_size)),label)))
    #         images.append(rimage)
    #     rawdata[cls[0]] = images
    # npimages = np.array(images)
    # np.random.shuffle(npimages)
    # rawdata[cls[0]] = npimages.T
# rawdata = np.array(rawdata)
# np.random.shuffle(rawdata)
# rawdata = rawdata.T
# data = rawdata[:image_size,:]
# labels = rawdata[image_size:,:]

# import pickle
# with open('training_data_dictionary.pkl','wb') as f:
#     pickle.dump(rawdata, f)

def select_subset(alldata, numsamples = 500):
    ret = []
    for key, val in alldata.iteritems():
        label = np.array([x == key for x in labelnames])
        rlabel = np.repeat(label[np.newaxis,:],numsamples,axis=0)
        pool = np.array(transform_images(val))
        choice = pool[np.random.choice(pool.shape[0], numsamples, replace=False),...].reshape(numsamples, pool.shape[1]*pool.shape[2])
        labeledchoice = np.concatenate((choice, rlabel),axis=1)
        ret.append(labeledchoice)
    return np.array(ret).reshape(len(alldata)*numsamples,-1).T

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

# seldata = select_subset(rawdata,500)
# data = seldata[:image_size,:]
# labels = seldata[image_size:,:]

ts = []
vs = []
