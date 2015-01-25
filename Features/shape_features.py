import numpy as np
from skimage import measure, morphology


#########################
###### TOOLS ############
#########################

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


####################
#### FEATURES ######
####################

def getMinorMajorRatio(image):
    image = image.copy()
    # Create the thresholded image to eliminate some of the background
    imagethr = np.where(image > np.mean(image),0.,1.0)

    #Dilate the image
    imdilated = morphology.dilation(imagethr, np.ones((4,4)))

    # Create the label list
    label_list = measure.label(imdilated)
    label_list = imagethr*label_list
    label_list = label_list.astype(int)
    
    region_list = measure.regionprops(label_list)
    maxregion = getLargestRegion(region_list, label_list, imagethr)
    
    # guard against cases where the segmentation fails by providing zeros
    ratio = 0.0
    if ((not maxregion is None) and  (maxregion.major_axis_length != 0.0)):
        ratio = 0.0 if maxregion is None else  maxregion.minor_axis_length*1.0 / maxregion.major_axis_length
    return ratio

def getHorizontalSymmetryFeature(image):
    image=image.copy()
    image2=np.fliplr(image.copy())
    return pseudoAutocorrelate(image, image2)

def getVerticalSymmetryFeature(image):
    image=image.copy()
    image2=np.flipud(image.copy())
    return pseudoAutocorrelate(image, image2)

def getCircularSymmetryFeature(image):
    image=image.copy()
    hs = getHorizontalSymmetryFeature(image)
    vs = getVerticalSymmetryFeature(image)
    image2=image.copy().T
    ds = pseudoAutocorrelate(image, image2)
    return np.power(hs*vs*ds, 1.0/3)

def getPercentPixelsAboveAverage(image):
    image = image.copy()
    image_threshold = image - np.mean(image)
    over_count = np.where(image_threshold > 0, 1.0, 0.0)
    return np.sum(over_count)/over_count.size

