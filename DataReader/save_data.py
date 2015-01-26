from skimage.io import imread
from skimage.transform import resize
import numpy as np
import os
import pickle
 
from training_list import classcounts
    
class DataLoader():
    def __init__(self, image_width=25, image_height=25):
        self.image_height=image_height
        self.image_width=image_width
        self.path = os.path.abspath(os.path.dirname(os.getcwd()))
        np.random.seed(1)

    def read_test_data(self):
        runsupervised = []
        filepath = os.path.join(self.path,"test")
        for filename in os.walk(filepath).next()[2]:
            if filename[-4:] != ".jpg":
              continue
            image = imread(os.path.join(filepath,filename), as_grey=True)
            rimage = resize(image, (self.image_height, self.image_width), order=1, mode="constant", cval=0.0)
            runsupervised.append(rimage)
        unsupervised = np.array(runsupervised)
        unsupervised = unsupervised.reshape(unsupervised.shape[0],625)

        with open('test_data_array.npy','wb') as f:
            np.save(f, unsupervised)

        with open('test_data_array.npy','rb') as f:
            return np.load(f)

    def read_training_data(self):
        rawdata = {}
        counter = 0
        for cls in classcounts:
            counter = counter + 1
            print "Importing " + cls[0], str(counter) + " of " + str(len(classcounts))
            filepath = os.path.join(self.path,"train",cls[0])
            images = []
            for filename in os.walk(filepath).next()[2]:
                if filename[-4:] != ".jpg":
                    continue
                image = imread(os.path.join(filepath,filename), as_grey=True)
                rimage = resize(image, (self.image_height, self.image_width), order=1, mode="constant", cval=0.0)
                images.append(rimage)
            rawdata[cls[0]] = images

        with open('training_data_dictionary.npy','wb') as f:
            np.save(f, rawdata)

        with open('training_data_dictionary.npy','rb') as f:
            return np.load(f)[()]
    
if __name__ == '__main__':
    dl = DataLoader()
    dl.read_training_data()
    dl.read_test_data()