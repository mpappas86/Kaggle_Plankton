import numpy as np
from skimage import measure, morphology
import os
from feature_tools import *
from annotations import *

@SHAPE
def getHeightFeature(images):
    return [getWhitespaceTrimmed(images[:,:,i]).shape[0] for i in xrange(images.shape[2])]

@SHAPE
def getLengthFeature(images):
    return [getWhitespaceTrimmed(images[:,:,i]).shape[1] for i in xrange(images.shape[2])]

@SHAPE
@MULT
def getMinorMajorRatioFeature(images):
    h = getHeightFeature(images)
    l = getLengthFeature(images)
    return [np.min([l[i], h[i]])*1.0/np.max([l[i], h[i]]) for i in xrange(images.shape[2])]

@SHAPE
def getHorizontalSymmetryFeature(images):
    return [pseudoAutocorrelate(images[:,:,i], np.fliplr(images[:,:,i])) for i in xrange(images.shape[2])]

@SHAPE
def getVerticalSymmetryFeature(images):
    return [pseudoAutocorrelate(images[:,:,i], np.flipud(images[:,:,i])) for i in xrange(images.shape[2])]

@SHAPE
def getTransposalSymmetryFeature(images):
    return [pseudoAutocorrelate(images[:,:,i], images[:,:,i].T) for i in xrange(images.shape[2])]

@SHAPE
def getCircularSymmetryFeature(images):
    hs = getHorizontalSymmetryFeature(images)
    vs = getVerticalSymmetryFeature(images)
    ts = getTransposalSymmetryFeature(images)
    return [np.power(hs[i]*vs[i]*ts[i], 1.0/3) for i in xrange(images.shape[2])]

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

