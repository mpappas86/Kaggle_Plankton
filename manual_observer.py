from skimage.io import imread
from skimage.transform import resize
import glob
import os
from matplotlib import pyplot as plt
from pylab import cm
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# get the classnames from the directory structure
path = '/Users/mikep/Desktop/Kaggle/Plankton'
directory_names = list(set(glob.glob(os.path.join(path,"train", "*"))\
 ).difference(set(glob.glob(os.path.join(path,"train","*.*")))))

# Generate training data
i = 0    

#get the total training images
numberofImages = 0
for folder in directory_names:
    for fileNameDir in os.walk(folder):   
        for fileName in fileNameDir[2]:
             # Only read in the images
            if fileName[-4:] != ".jpg":
              continue
            numberofImages += 1

#Note that numImages at once must be cleanly divisible by 5 based on the current format.
numRowsAtOnce = 5
numColumnsAtOnce = 5
numImagesAtOnce = numRowsAtOnce*numColumnsAtOnce
images = [0]*numImagesAtOnce
nIm = 0
justShowed = ""

def preProcessImages(image):
    return resize(image, (25, 25))

print "Reading images"
# Navigate through the list of directories
for folder in directory_names:
    # Append the string class name for each class
    currentClass = folder.split(os.pathsep)[-1]
    for fileNameDir in os.walk(folder):   
        for fileName in fileNameDir[2]:
            # Only read in the images
            if fileName[-4:] != ".jpg":
              continue
            
            # Read in the images and create the features
            nameFileImage = "{0}{1}{2}".format(fileNameDir[0], os.sep, fileName)            
            image = preProcessImages(imread(nameFileImage, as_grey=True))
            if(not justShowed==currentClass):
                images[nIm] = image
                nIm += 1
                if(nIm==numImagesAtOnce):
                    justShowed = currentClass
                    nIm = 0
                    fig, axes = plt.subplots(numRowsAtOnce, numColumnsAtOnce)
                    axes = axes.ravel()
                    for j in range(0, numImagesAtOnce):
                        ax = axes[j]
                        ax.imshow(images[j], cmap = cm.Greys_r)
                    fig.suptitle(currentClass)
                    plt.show()


            i += 1
            # report progress for each 5% done  
            report = [int((j+1)*numberofImages/20.) for j in range(20)]
            if i in report: print np.ceil(i *100.0 / numberofImages), "% done"