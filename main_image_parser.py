from skimage import data, io, filter
import os
import pre_data
import numpy as np

train_img_path = '/Users/mikep/Desktop/Kaggle/Plankton/train'
test_img_path = '/Users/mikep/Desktop/Kaggle/Plankton/test'
#test_collection = io.imread_collection(os.path.join(test_img_path, '*.jpg'))

#Unfortunately all the images are different sizes, so we can't directly do
#something like "probability this pixel is white/black."

#My first crack is going to be something super raw. My vector will be [average, median, std]
#of pixel values. Obviously these won't differ all that much, but I just want to get a 
#classifying system working.

linear_classifier = [[0, 0, 0, 0] for a_class in pre_data.classifications]
learning_rate = 0.1
error_threshold = .5
ix = 0

#For the training images for each classification...
for a_class in pre_data.classifications:
  #Collect all the images.
  train_collection = io.imread_collection(os.path.join(train_img_path, a_class, "*.jpg"))
  num_images = len(train_collection)
  classifier_ix = 0
  #Then, for each classifier...
  for cur_classifier in linear_classifier:
    #If the classifier corresponds to the current class, expect to pass. Else expect to fail.
    if classifier_ix == ix:
      expected_result = 1
    else:
      expected_result = 0

    num_errors = 0.0
    while True:
      #For each image, perform the perceptron algorithm.
      for the_image in train_collection:
        image = np.array(the_image)/255
        #Compute feature vector.
        feature_vec = np.array([1, np.average(image), np.median(image), np.std(image)])
        #Run perceptron.
        percep_result = np.dot(feature_vec, cur_classifier)
        #Compute normed error, i.e whether we passed or failed.
        normed_error = np.abs(expected_result-percep_result)
        normed_error = normed_error if normed_error == 1 else 0
        #Update classifier.
        cur_classifier = cur_classifier + learning_rate*normed_error*feature_vec
        print(str(percep_result) + " " + str(normed_error) + " " + str(cur_classifier))
        #Update worst error.
        if(normed_error == 0):
          num_errors = num_errors + 1

      #If we exceed the error threshold, keep looping until we don't.
      if(num_errors/num_images < error_threshold):
        break

    #Update classifier in the overall cache of classifier info.
    linear_classifier[classifier_ix] = cur_classifier
    classifier_ix = classifier_ix + 1
  #Update which class we're on.
  ix = ix + 1

