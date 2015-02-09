import numpy as np
from skimage import measure, morphology
import os
from feature_tools import getLargestRegion, pseudoAutocorrelate
from annotations import *

#@FILE
#@MULT
def getCompressionRatioFeature(images):
    tmp_file_uncompressed = "tmp.npz"
    tmp_file_compressed = "tmp2.npz"
    feature_vals = []
    for i in xrange(images.shape[2]):
        try:
            image = images[:,:,i]
            np.savez(tmp_file_uncompressed, image)
            np.savez_compressed(tmp_file_compressed, image)
            uc_size = os.path.getsize(tmp_file_uncompressed)
            c_size = os.path.getsize(tmp_file_compressed)
            feature_vals.append(1.0*c_size/uc_size)
        finally:
            os.remove(tmp_file_compressed)
            os.remove(tmp_file_uncompressed)
    return feature_vals


#The below 3 features currently are useless because we only calculate features on rescaled images.
#Possible solutions are to pass the filename along with the data, compute these features before the
#original rescaling, or simply drop these features.

#@FILE
#def getImageFileHeightFeature(images):  
#    return [images[:,:,i].shape[0] for i in xrange(images.shape[2])]

#@FILE
#def getImageFileLengthFeature(images):
#    return [images[:,:,i].shape[1] for i in xrange(images.shape[2])]

#@FILE
#@MULT
#def getImageFileMinorMajorRatioFeature(images):
#    def npmin(image):
#        return np.min([image.shape[1], image.shape[0]])
#    def npmax(iamge):
#        return np.max([image.shape[1], image.shape[0]])
#    return [npmin(images[:,:,i])*1.0/npmin(images[:,:,i]) for i in xrange(images.shape[2])]

