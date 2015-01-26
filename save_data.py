from skimage.io import imread
from skimage.transform import resize
import numpy as np
import os
import pickle

from training_list import classcounts
path = os.getcwd()
np.random.seed(1)

image_width = 25
image_height = 25
image_size = image_width*image_height

# rawdata = {}
# counter = 0
# for cls in classcounts:
#     counter = counter + 1
#     print "Importing " + cls[0], str(counter) + " of " + str(len(classcounts))
#     filepath = path + "/train/" + cls[0] + "/"
#     images = []
#     for filename in os.walk(filepath).next()[2]:
#         image = imread(filepath+filename, as_grey=True)
#         rimage = resize(image, (image_height, image_width), order=1, mode="constant", cval=0.0)
#         images.append(rimage)
#     rawdata[cls[0]] = images

# with open('training_data_dictionary.npy','wb') as f:
#     np.save(f, rawdata)

# with open('training_data_dictionary.npy','rb') as f:
#     rawdata = np.load(f)[()]
    
# runsupervised = []
# filepath = path+"/test/"
# for filename in os.walk(filepath).next()[2]:
#     image = imread(filepath+filename, as_grey=True)
#     rimage = resize(image, (image_height, image_width), order=1, mode="constant", cval=0.0)
#     runsupervised.append(rimage)
# unsupervised = np.array(runsupervised)
# unsupervised = unsupervised.reshape(unsupervised.shape[0],625)

# with open('test_data_array.npy','wb') as f:
#     np.save(f, unsupervised)

# with open('test_data_array.npy','rb') as f:
#     unsupervised = np.load(f)
