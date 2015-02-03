import numpy as np
from layer import Layer

class Neural_Layer(Layer):
    def __init__(self, input_size, output_size, order=None):
        super(Neural_Layer, self).__init__(input_size, output_size, order=order)
        self.df = lambda x: 0*x+1

    # Each row of the data is a feature, and each column is a sample
    def training_predict(self, data, dropout_in, dropout_array, node):
        self.batch_size = float(data.shape[1])
        node.dataup = data[dropout_in,:]
        ii = np.where(dropout_in)[0]
        oi = np.where(dropout_array)[0][:,np.newaxis]
        node.savedup = self.W[oi,ii].dot(node.dataup)+self.b[dropout_array]
        return self.f(node.savedup)

    def backprop_setup(self):
        self.Wgrad = np.zeros(self.Wgrad.shape)
        self.bgrad = np.zeros(self.bgrad.shape)        
    
    def backprop(self, upstream, dropout_in, dropout_array, node):
        delta = self.df(node.savedup)*upstream[dropout_array,:]
        ii = np.where(dropout_in)[0]
        oi = np.where(dropout_array)[0][:,np.newaxis]
        self.Wgrad[oi,ii] += delta.dot(node.dataup.T)
        self.bgrad[dropout_array] += delta.sum(1)[:,np.newaxis]
        self.updated_yet = False
        node.savedup = None
        node.dataup = None
        return self.W[oi,ii].T.dot(delta)

    def set_df(self, df):
        self.df = df
