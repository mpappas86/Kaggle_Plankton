from skimage.io import imread
from skimage.transform import resize
import numpy as np
import os
np.random.seed(2)

from training_list import classcounts
import grading

num_samples = sum([x[1] for x in classcounts])
num_labels = len(classcounts)

labels = []
counter = 0
for cls in classcounts:
    labels.extend([counter for y in xrange(cls[1])])
    counter = counter + 1
labels = np.array(labels)

random_guess = np.zeros((num_samples, num_labels))
random_guess[range(num_samples),np.random.randint(0,num_labels,num_samples)] = 1
random_llos = grading.multiclass_log_loss(labels, random_guess)

arbitrary_prior = np.random.rand(num_labels)
arbitrary_prior = arbitrary_prior/arbitrary_prior.sum()
arbitrary_guess = np.repeat(arbitrary_prior[:,np.newaxis],num_samples,axis=1).T
arbitrary_llos = grading.multiclass_log_loss(labels, arbitrary_guess)

uniform_guess = np.ones((num_samples, num_labels))/float(num_labels)
uniform_llos = grading.multiclass_log_loss(labels, uniform_guess)

frequency_prior = np.array([cls[1]/float(num_samples) for cls in classcounts])
frequency_guess = np.repeat(frequency_prior[:,np.newaxis],num_samples,axis=1).T
frequency_llos = grading.multiclass_log_loss(labels, frequency_guess)

most_common_prior = np.zeros(num_labels)
most_common_prior[np.array([x[1] for x in classcounts]).argmax()] = 1
most_common_guess = np.repeat(most_common_prior[:,np.newaxis],num_samples,axis=1).T
most_common_llos = grading.multiclass_log_loss(labels, most_common_guess)
