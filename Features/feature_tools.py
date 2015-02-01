import numpy as np
import os
from annotations import *

def getFeatures(annotation_list=[]):
    if annotation_list==[]:
        annotation_list=ALL
    features = set()
    feature_vals = []
    for a in annotation_list:
        features = features.union(set(a.all.values()))
    return features

def getWhitespaceTrimmed(image):
    newimage = None
    for r in image:
        if (np.sum(r)*1.0/len(r) < 255):
            if(newimage is None):
                newimage = r
            else:
                newimage = np.vstack((newimage, r))
    newimage=newimage.T
    finalimage=None
    for c in newimage:
        if (np.sum(c)*1.0/len(c) < 255):
            if(finalimage is None):
                finalimage = c
            else:
                finalimage = np.vstack((finalimage, c))
    return finalimage.T

# find the largest nonzero region
def getLargestRegion(props, labelmap, imagethres):
    regionmaxprop = None
    for regionprop in props:
        # check to see if the region is at least 50% nonzero
        if sum(imagethres[labelmap == regionprop.label])*1.0/regionprop.area < 0.50:
            continue
        if regionmaxprop is None:
            regionmaxprop = regionprop
        if regionmaxprop.filled_area < regionprop.filled_area:
            regionmaxprop = regionprop
    return regionmaxprop

def pseudoAutocorrelate(image1, image2):
    image1=image1.flatten()
    image2=image2.flatten()
    
    #Lock values between 0 and 1
    max_val = np.max(image1)
    image1 = image1*1.0/max_val
    image2 = image2*1.0/max_val

    def closeMatch(a, b):
        div = a/b
        if((div >.8 and div < 1.2) or (a<0.05 and b<0.05)):
            return True
        return False

    result = [1 if closeMatch(image1[i], image2[i]) else 0 for i in range(0, len(image1))]
    result = np.mean(result)

    return result