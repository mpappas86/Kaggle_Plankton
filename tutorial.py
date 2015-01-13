from skimage.io import imread
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier as RF
import glob
import os
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from matplotlib import colors
from pylab import cm
from skimage import segmentation, measure, morphology
from skimage.morphology import watershed
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.feature import peak_local_max
import shape_features as sf
import grading
# make graphics inline
#%matplotlib inline
import warnings
warnings.filterwarnings("ignore")

# get the classnames from the directory structure
path = '/Users/mikep/Desktop/Kaggle/Plankton'
directory_names = list(set(glob.glob(os.path.join(path,"train", "*"))\
 ).difference(set(glob.glob(os.path.join(path,"train","*.*")))))

#get the total training images
numberofImages = 0
for folder in directory_names:
    for fileNameDir in os.walk(folder):   
        for fileName in fileNameDir[2]:
             # Only read in the images
            if fileName[-4:] != ".jpg":
              continue
            numberofImages += 1

# We'll rescale the images to be 25x25
maxPixel = 25
imageSize = maxPixel * maxPixel
num_rows = numberofImages # one row for each image in the training dataset
num_features = imageSize + 2 # for our ratio

# X is the feature vector with one row of features per image
# consisting of the pixel values and our metric
X = np.zeros((num_rows, num_features), dtype=float)
# y is the numeric class label 
y = np.zeros((num_rows))

# Generate training data
i = 0    
label = 0
# List of string of class names
namesClasses = list()

print "Reading images"
# Navigate through the list of directories
for folder in directory_names:
    # Append the string class name for each class
    currentClass = folder.split(os.pathsep)[-1]
    namesClasses.append(currentClass)
    for fileNameDir in os.walk(folder):   
        for fileName in fileNameDir[2]:
            # Only read in the images
            if fileName[-4:] != ".jpg":
              continue
            
            # Read in the images and create the features
            nameFileImage = "{0}{1}{2}".format(fileNameDir[0], os.sep, fileName)            
            image = imread(nameFileImage, as_grey=True)
            axisratio = sf.getMinorMajorRatio(image)
            pixels_above_mean = sf.getPercentPixelsAboveAverage(image)
            image = resize(image, (maxPixel, maxPixel))
            
            # Store the rescaled image pixels and the axis ratio
            X[i, 0:imageSize-1] = np.reshape(image, (1, imageSize))
            X[i, imageSize-1] = axisratio
            X[i, imageSize] = pixels_above_mean
            
            # Store the classlabel
            y[i] = label
            i += 1
            # report progress for each 5% done  
            report = [int((j+1)*num_rows/20.) for j in range(20)]
            if i in report: print np.ceil(i *100.0 / num_rows), "% done"
    label += 1

print "Training"
# n_estimators is the number of decision trees
# max_features also known as m_try is set to the default value of the square root of the number of features
clf = RF(n_estimators=100, n_jobs=3);
scores = cross_validation.cross_val_score(clf, X, y, cv=5, n_jobs=1);
print "Accuracy of all classes"
print np.mean(scores)

# Get the probability predictions for computing the log-loss function
kf = KFold(y, n_folds=5)
# prediction probabilities number of samples, by number of classes
y_prob_pred = np.zeros((len(y),len(set(y))))
y_best_pred = y*0
for train, test in kf:
    X_train, X_test, y_train, y_test = X[train,:], X[test,:], y[train], y[test]
    clf = RF(n_estimators=100, n_jobs=3)
    clf.fit(X_train, y_train)
    y_best_pred[test] = clf.predict(X_test)
    y_prob_pred[test] = clf.predict_proba(X_test)

print classification_report(y, y_pred, target_names=namesClasses)
print(grading.multiclass_log_loss(y, y_pred))





#EXAMPLE CODE FOR PLOTTING THINGS AGAINST EACH OTHER, NOT NECESSARY FOR RUN!

# Loop through the classes two at a time and compare their distributions of the Width/Length Ratio

# #Create a DataFrame object to make subsetting the data on the class 
# df = pd.DataFrame({"class": y[:], "ratio": X[:, num_features-1]})

# f = plt.figure(figsize=(30, 20))
# #we suppress zeros and choose a few large classes to better highlight the distributions.
# df = df.loc[df["ratio"] > 0]
# minimumSize = 20 
# counts = df["class"].value_counts()
# largeclasses = [int(x) for x in list(counts.loc[counts > minimumSize].index)]
# # Loop through 40 of the classes 
# for j in range(0,40,2):
#     subfig = plt.subplot(4, 5, j/2 +1)
#     # Plot the normalized histograms for two classes
#     classind1 = largeclasses[j]
#     classind2 = largeclasses[j+1]
#     n, bins,p = plt.hist(df.loc[df["class"] == classind1]["ratio"].values,\
#                          alpha=0.5, bins=[x*0.01 for x in range(100)], \
#                          label=namesClasses[classind1].split(os.sep)[-1], normed=1)

#     n2, bins,p = plt.hist(df.loc[df["class"] == (classind2)]["ratio"].values,\
#                           alpha=0.5, bins=bins, label=namesClasses[classind2].split(os.sep)[-1],normed=1)
#     subfig.set_ylim([0.,10.])
#     plt.legend(loc='upper right')
#     plt.xlabel("Width/Length Ratio")

