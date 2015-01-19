import numpy as np
import copy
from neural_layer import Neural_Layer

class Softplus_Layer(Neural_Layer):
    def __init__(self, input_size, output_size):
        super(Softplus_Layer, self).__init__(input_size, output_size)
        self.f = self.softplus
        #TODO:: Confirm that this is the correct df. The current value is df = d(softplus(z))/dz, expressed with x=softplus(z).
        self.df = lambda x: (np.exp(x)-1)/(np.exp(x))

    def softplus(self, data):
        return np.log(1.0 + np.exp(data))

    # TODO:: Verify that this is correct - it's unchanged from the sigmoid case currently.
    # We only need the simgoid output in backprop, so save that instead of saving just the linear function
    def predict(self, data):
        self.batch_size = float(data.shape[1])
        self.dataup = data
        self.savedup = self.f(self.W.dot(self.dataup)+self.b)
        return copy.copy(self.savedup)
