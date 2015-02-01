from manual_observer import ManualObserver
import Features.shape_features as sf
import types
from matplotlib import pyplot as plt
from pylab import cm

def patch_method(target_object):
    def preProcessImages(target_object,image):
        return sf.getHeightFeature(image)
    target_object.preProcessImages = types.MethodType(preProcessImages,target_object)

mo = ManualObserver()

patch_method(mo)

mo.checkOneProcessedImage(view=True)
