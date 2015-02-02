import numpy as np
import copy
from neural_layer import Neural_Layer

class Sigmoid_Layer(Neural_Layer):
    def __init__(self, input_size, output_size, order=None):
        super(Sigmoid_Layer, self).__init__(input_size, output_size, order=order)
        self.f = self.sigmoid
        # df is to be used with savedup, which is the simgoided input
        self.df = lambda x: x*(1-x)

    def sigmoid(self, data):
        return 1.0 / (1.0 + np.exp(-data))

    # We only need the simgoid output in backprop, so save that instead of saving just the linear function
    # def predict(self, data, node):
    #     self.batch_size = float(data.shape[1])
    #     node.dataup = copy.copy(data)
    #     node.savedup = self.f(self.W.dot(node.dataup)+self.b)
    #     return copy.copy(node.savedup)

    def training_predict(self, data, dropout_in, dropout_array, node):
        self.batch_size = float(data.shape[1])
        node.dataup = copy.copy(data[dropout_in,:])
        ii = np.where(dropout_in)[0]
        oi = np.where(dropout_array)[0][:,np.newaxis]
        node.savedup = self.f(self.W[oi,ii].dot(node.dataup)+self.b[dropout_array])
        return copy.copy(node.savedup)
