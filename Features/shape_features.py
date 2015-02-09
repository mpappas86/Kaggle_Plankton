import numpy as np
from skimage import measure, morphology
import os
from feature_tools import *
from annotations import *

#The below section serves as an effective cache for whitespace trimmed images so we don't recompute
#several times.
############################
recent_images = np.array([])
trimmed_images = np.array([])

def load_to_recent_images(images):
    global recent_images
    global trimmed_images
    if(np.array(recent_images==images).all()):
        return True
    else:
        recent_images=images
        trimmed_images=np.array([getWhitespaceTrimmed(images[:,:,i]) for i in xrange(images.shape[2])]).T
        return False
#############################

@SHAPE
def getHeightFeature(images):
    if(load_to_recent_images(images)):
        return [trimmed_images[:,:,i].shape[0] for i in xrange(recent_images.shape[2])]
    return [getWhitespaceTrimmed(images[:,:,i]).shape[0] for i in xrange(images.shape[2])]

@SHAPE
def getLengthFeature(images):
    if(load_to_recent_images(images)):
        return [trimmed_images[:,:,i].shape[1] for i in xrange(recent_images.shape[2])]
    return [getWhitespaceTrimmed(images[:,:,i]).shape[1] for i in xrange(images.shape[2])]

@SHAPE
@MULT
def getMinorMajorRatioFeature(images):
    h = np.array(getHeightFeature(images))
    l = np.array(getLengthFeature(images))
    return np.min([l, h], axis=0)*1.0/np.max([l, h], axis=0)
    
@SHAPE
def getHorizontalSymmetryFeature(images):
    def computeVal(image):
        return pseudoAutocorrelate(image, np.fliplr(image))
    return [computeVal(images[:,:,i]) for i in xrange(images.shape[2])]

@SHAPE
def getVerticalSymmetryFeature(images):
    def computeVal(image):
        return pseudoAutocorrelate(image, np.flipud(image))
    return [computeVal(images[:,:,i]) for i in xrange(images.shape[2])]
    
@SHAPE
def getTransposalSymmetryFeature(images):
    def computeVal(image):
        return pseudoAutocorrelate(image, image.T)
    return [computeVal(images[:,:,i]) for i in xrange(images.shape[2])]

@SHAPE
def getCircularSymmetryFeature(images):
    hs = np.array(getHorizontalSymmetryFeature(images))
    vs = np.array(getVerticalSymmetryFeature(images))
    ts = np.array(getTransposalSymmetryFeature(images))
    return np.power(hs*vs*ts, 1.0/3)

@SHAPE
def getPercentPixelsAboveAverage(images):
    def calcPercent(image):
        return np.sum(np.where(image - np.mean(image) > 0, 1.0, 0.0))/image.size
    return [calcPercent(images[:,:,i]) for i in xrange(images.shape[2])]

# def getMinorMajorRatioFeature(image):
#     image = image.copy()
#     # Create the thresholded image to eliminate some of the background
#     imagethr = np.where(image > np.mean(image),0.,1.0)

#     #Dilate the image
#     imdilated = morphology.dilation(imagethr, np.ones((4,4)))

#     # Create the label list
#     label_list = measure.label(imdilated)
#     label_list = imagethr*label_list
#     label_list = label_list.astype(int)
    
#     region_list = measure.regionprops(label_list)
#     maxregion = getLargestRegion(region_list, label_list, imagethr)
    
#     # guard against cases where the segmentation fails by providing zeros
#     ratio = 0.0
#     if ((not maxregion is None) and  (maxregion.major_axis_length != 0.0)):
#         ratio = 0.0 if maxregion is None else  maxregion.minor_axis_length*1.0 / maxregion.major_axis_length
#     return ratio

