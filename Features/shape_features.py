import numpy as np
from skimage import measure, morphology
import os
from feature_tools import getLargestRegion, pseudoAutocorrelate, getWhitespaceTrimmed
from annotations import *

@SHAPE
def getHeightFeature(image):
    image = getWhitespaceTrimmed(image.copy())
    return image.shape[0]

@SHAPE
def getLengthFeature(image):
    image = getWhitespaceTrimmed(image.copy())
    return image.shape[1]

@SHAPE
@MULT
def getMinorMajorRatioFeature(image):
    image = image.copy()
    h = getHeightFeature(image)
    l = getLengthFeature(image)
    return np.min([l, h])*1.0/np.max([l, h])

@SHAPE
def getHorizontalSymmetryFeature(image):
    image=image.copy()
    image2=np.fliplr(image.copy())
    return pseudoAutocorrelate(image, image2)

@SHAPE
def getVerticalSymmetryFeature(image):
    image=image.copy()
    image2=np.flipud(image.copy())
    return pseudoAutocorrelate(image, image2)

@SHAPE
def getTransposalSymmetryFeature(image):
    image=image.copy()
    image2=image.copy().T
    return pseudoAutocorrelate(image, image2)

@SHAPE
def getCircularSymmetryFeature(image):
    image=image.copy()
    hs = getHorizontalSymmetryFeature(image)
    vs = getVerticalSymmetryFeature(image)
    ts = getTransposalSymmetryFeature(image)
    return np.power(hs*vs*ts, 1.0/3)

@SHAPE
def getPercentPixelsAboveAverage(image):
    image = image.copy()
    image_threshold = image - np.mean(image)
    over_count = np.where(image_threshold > 0, 1.0, 0.0)
    return np.sum(over_count)/over_count.size

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

