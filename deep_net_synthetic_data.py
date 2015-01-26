from DataReader.read_images import read_training

import plankton_nnet

# Min = 9, Max = 1979

image_width = 25
image_height = 25
image_size = image_width*image_height

rawdata = read_training(image_width, image_height)

nnet, gfile = plankton_nnet.build(image_size, glrate=0.01)

plankton_nnet.deploy(rawdata, nnet)