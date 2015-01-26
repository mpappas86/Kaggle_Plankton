import save_data
import numpy as np

def read_training(image_width=25, image_height=25):
  try:
      len(rawdata)
      assert(rawdata.values()[0][0].shape[0] == image_width)
      assert(rawdata.values()[0][0].shape[1] == image_height)
  except:
      try:
          with open('training_data_dictionary.npy','rb') as f:
              rawdata = np.load(f)[()]
              assert(rawdata.values()[0][0].shape[0] == image_width)
              assert(rawdata.values()[0][0].shape[1] == image_height)
      except:
          raise
          dl = save_data.DataLoader(image_width=image_width, image_height=image_height)
          rawdata = dl.read_training_data()
  return rawdata

def read_test(image_width=25, image_height=25):
  try:
      len(unsupervised)
      assert(unsupervised[0][0].shape[0] == image_width)
      assert(unsupervised[0][0].shape[1] == image_height)
  except:
      try:
          with open('test_data_array.npy','rb') as f:
              unsupervised = np.load(f)[()]
              assert(unsupervised[0][0].shape[0] == image_width)
              assert(unsupervised[0][0].shape[1] == image_height)
      except:
          dl = save_data.DataLoader(image_width=image_width, image_height=image_height)
          unsupervised = dl.read_test_data()
  return unsupervised
