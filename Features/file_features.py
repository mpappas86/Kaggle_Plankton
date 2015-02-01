import numpy as np
from skimage import measure, morphology
import os
from feature_tools import getLargestRegion, pseudoAutocorrelate
from annotations import *

@FILE
@MULT
def getCompressionRatioFeature(image):
    tmp_file_uncompressed = "tmp.npz"
    tmp_file_compressed = "tmp2.npz"
    try:
        np.savez(tmp_file_uncompressed, image)
        np.savez_compressed(tmp_file_compressed, image)
        uc_size = os.path.getsize(tmp_file_uncompressed)
        c_size = os.path.getsize(tmp_file_compressed)
    finally:
        os.remove(tmp_file_compressed)
        os.remove(tmp_file_uncompressed)
    return 1.0*c_size/uc_size

@FILE
def getImageFileHeightFeature(image):  
    return image.shape[0]

@FILE
def getImageFileLengthFeature(image):
    return image.shape[1]

@FILE
@MULT
def getImageFileMinorMajorRatioFeature(image):
    h = image.shape[0]
    l = image.shape[1]
    return np.min([l, h])*1.0/np.max([l, h])

