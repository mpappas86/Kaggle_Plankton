import matplotlib
matplotlib.use('Agg')
from skimage.io import imread
from skimage.transform import resize
import glob
import os
from matplotlib import pyplot as plt
from pylab import cm, savefig
import numpy as np
import warnings
warnings.filterwarnings("ignore")

#WARNING - PNGs are currently saved to the individual class dirs rather than ClassImages due to a bug.
#currentClass is actually the full path to the class folder, not the class name.
#I could fix this but frankly am too annoyed at this right now to bother...

class ManualObserver():
    def __init__(self):
        # get the classnames from the directory structure
        path = '/Users/mikep/Desktop/Kaggle/Plankton'
        self.directory_names = list(set(glob.glob(os.path.join(path,"train", "*"))\
         ).difference(set(glob.glob(os.path.join(path,"train","*.*")))))
   
        #get the total training images
        self.numberofImages = 0
        for folder in self.directory_names:
            for fileNameDir in os.walk(folder):   
                for fileName in fileNameDir[2]:
                     # Only read in the images
                    if fileName[-4:] != ".jpg":
                      continue
                    self.numberofImages += 1

        self.numRowsAtOnce = 5
        self.numColumnsAtOnce = 5
        self.numImagesAtOnce = self.numRowsAtOnce*self.numColumnsAtOnce

    def preProcessImages(self, image):
        return resize(image, (25, 25))

    def checkOneProcessedImage(self, view=False):
        print "Grabbing an image"
        for folder in self.directory_names:
            for fileNameDir in os.walk(folder):
                for fileName in fileNameDir[2]:
                    if fileName[-4:] == ".jpg":
                        nameFileImage = "{0}{1}{2}".format(fileNameDir[0], os.sep, fileName)
                        image = imread(nameFileImage, as_grey=True)
                        if(view):
                            plt.imshow(image, cmap=cm.Greys_r)
                            plt.show()
                        return self.preProcessImages(image)

    def runThrough(self):
        print "Reading images"
        i = 0 
        images = [0]*self.numImagesAtOnce
        nIm = 0
        justShowed = ""
        # Navigate through the list of directories
        for folder in self.directory_names:
            # Append the string class name for each class
            currentClass = folder.split(os.pathsep)[-1]
            for fileNameDir in os.walk(folder):   
                for fileName in fileNameDir[2]:
                    # Only read in the images
                    if fileName[-4:] != ".jpg":
                      continue
                    
                    # Read in the images and create the features
                    nameFileImage = "{0}{1}{2}".format(fileNameDir[0], os.sep, fileName)            
                    image = self.preProcessImages(imread(nameFileImage, as_grey=True))
                    if(not justShowed==currentClass):
                        images[nIm] = image
                        nIm += 1
                        if(nIm==self.numImagesAtOnce):
                            justShowed = currentClass
                            nIm = 0
                            fig, axes = plt.subplots(self.numRowsAtOnce, self.numColumnsAtOnce)
                            axes = axes.ravel()
                            for j in range(0, self.numImagesAtOnce):
                                ax = axes[j]
                                ax.imshow(images[j], cmap = cm.Greys_r)
                            fig.suptitle(currentClass)
                            curdir = os.path.dirname(os.path.realpath(__file__))
                            fig.savefig(os.path.join(curdir, "ClassImages", "{0}.png".format(currentClass)))
                            #plt.show()


                    i += 1
                    # report progress for each 5% done  
                    report = [int((j+1)*self.numberofImages/20.) for j in range(20)]
                    if i in report: print np.ceil(i *100.0 / self.numberofImages), "% done"


if __name__ == "__main__":
    mo = ManualObserver()
    mo.runThrough()


