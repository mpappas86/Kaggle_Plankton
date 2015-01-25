from manual_observer import ManualObserver
import Features.shape_features as sf
import types

def patch_method(target_object):
    def preProcessImages(target_object,image):
        print(sf.getHorizontalSymmetryFeature(image))
        print(sf.getVerticalSymmetryFeature(image))
        print(sf.getCircularSymmetryFeature(image))
    target_object.preProcessImages = types.MethodType(preProcessImages,target_object)

mo = ManualObserver()

patch_method(mo)

mo.checkOneProcessedImage(view=False)
