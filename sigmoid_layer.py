import numpy as np
import copy
from neural_layer import Neural_Layer

class Sigmoid_Layer(Neural_Layer):
    def __init__(self, input_size, output_size):
        super(Sigmoid_Layer, self).__init__(input_size, output_size)
        self.f = self.sigmoid
        # df is to be used with savedup, which is the simgoided input
        self.df = lambda x: x*(1-x)

    def sigmoid(self, data):
        return 1.0 / (1.0 + np.exp(-data))

    # We only need the simgoid output in backprop, so save that instead of saving just the linear function
    def predict(self, data, node):
        self.batch_size = float(data.shape[1])
        node.dataup = copy.copy(data)
        node.savedup = self.f(self.W.dot(node.dataup)+self.b)
        return copy.copy(node.savedup)
