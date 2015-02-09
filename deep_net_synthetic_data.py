from DataReader.read_images import read_training
from Features.feature_tools import getFeatures

#This is a bit sad, but basically this is a trick to make sure that all annotated features
#are detected by modules that just import annotations.py. All feature files will need to be
#added to this list to be recognized.
import Features.file_features
import Features.shape_features

from Features.annotations import *

import plankton_nnet

# Min = 9, Max = 1979

image_width = 25
image_height = 25
image_size = image_width*image_height

rawdata = read_training(image_width, image_height)

feature_list=getFeatures()
input_size=image_size+len(feature_list)

nnet, gfile = plankton_nnet.build(input_size, glrate=0.01)

plankton_nnet.deploy(rawdata, (image_height, image_width), nnet, feature_list=feature_list)
