from mllib import deploy_nnet

#This file should be where we translate from the totally general deploy_nnet to the much less general
#use case of the particular project.

def build(image_size, glrate=0.01):
  return deploy_nnet.build(image_size, glrate)

def deploy(rawdata, image_size, nnet, num_epochs=10, feature_list=None):
  deploy_nnet.deploy(rawdata, image_size, nnet, num_epochs, feature_list)