import numpy as np
import copy

class Maxpool_Layer(object):
    def __init__(self, input_size):
        self.shape = (1, input_size)
        self.batch_size = None
        self.updated_yet = True

    def predict(self, data, node):
        node.savedup = np.argmax(data,0)
        return np.max(data,0)[np.newaxis,:]
        
    def backprop(self, upstream, node):
        delta = np.zeros((self.shape[1],upstream.shape[1]))
        for elnum in xrange(upstream.shape[0]):
            delta[node.savedup[elnum], elnum] = upstream[0,elnum]
        return delta

    def backprop_setup(self):
        pass

    def update_weights(self):
        pass

    def get_num_params(self, *args, **kwargs):
        pass

    def set_weights(self, *args, **kwargs):
        pass

    def get_weights(self, *args, **kwargs):
        pass

    def set_weight_vector(self, *args, **kwargs):
        pass

    def get_weight_vector(self, *args, **kwargs):
        pass

    def set_weights(self, *args, **kwargs):
        pass

    def get_weights(self, *args, **kwargs):
        pass

    def set_lrates(self, *args, **kwargs):
        pass

    def set_moments(self, *args, **kwargs):
        pass

    def set_weight_decay(self, *args, **kwargs):
        pass

    def set_f(self, *args, **kwargs):
        pass

    def set_df(self, *args, **kwargs):
        pass
