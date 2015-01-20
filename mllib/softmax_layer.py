import numpy as np
import copy
from neural_layer import Neural_Layer

class Softmax_Layer(Neural_Layer):
    def __init__(self, input_size, output_size, temperature):
        super(Softmax_Layer, self).__init__(input_size, output_size)
        self.temperature = temperature
        self.f = self.softmax
        self.df = self.dsoftmax

    # Each row of the data is a feature, and each column is a sample
    def predict(self, data, node):
        self.batch_size = float(data.shape[1])
        node.dataup = data
        node.savedup = self.f(self.W.dot(node.dataup)+self.b)
        return copy.copy(node.savedup)

    def softmax(self, data):
        tmp = np.exp(data / self.temperature)
        return tmp / tmp.sum(0)

    def dsoftmax(self, data):
        return (np.diag(data) - np.outer(data, data))/self.temperature
    
    def gen_delta(self, upstream, savedup):
        delta = np.zeros(savedup.shape)
        for col in xrange(savedup.shape[1]):
            delta[:,col] = self.df(savedup[:,col]).dot(upstream[:,col])
        return delta
    
    def backprop(self, upstream, node):
        delta = self.gen_delta(upstream, node.savedup)
        self.Wgrad += delta.dot(node.dataup.T)
        self.bgrad += delta.sum(1)[:,np.newaxis]
        self.updated_yet = False
        node.savedup = None
        node.dataup = None
        return self.W.T.dot(delta)
