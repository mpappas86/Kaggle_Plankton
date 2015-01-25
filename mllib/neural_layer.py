import numpy as np
from layer import Layer

class Neural_Layer(Layer):
    def __init__(self, input_size, output_size):
        super(Neural_Layer, self).__init__(input_size, output_size)
        self.df = lambda x: 0*x+1
        self.updated_yet = True

    # Each row of the data is a feature, and each column is a sample
    def predict(self, data, node):
        self.batch_size = float(data.shape[1])
        node.dataup = data
        node.savedup = self.W.dot(node.dataup)+self.b
        return self.f(node.savedup)

    def backprop_setup(self):
        self.Wgrad = np.zeros(self.Wgrad.shape)
        self.bgrad = np.zeros(self.bgrad.shape)        
    
    def backprop(self, upstream, node):
        delta = self.df(node.savedup)*upstream
        self.Wgrad += delta.dot(node.dataup.T)
        self.bgrad += delta.sum(1)[:,np.newaxis]
        self.updated_yet = False
        node.savedup = None
        node.dataup = None
        return self.W.T.dot(delta)

    def update_weights(self):
        if not self.updated_yet:
            self.Wspeed = self.nu*self.Wspeed - (self.lr*self.lam)*self.W - self.lr/self.batch_size*self.Wgrad
            self.bspeed = self.nub*self.bspeed - self.lrb/self.batch_size*self.bgrad
            self.W += self.Wspeed
            self.b += self.bspeed
            self.updated_yet = True

    def set_df(self, df):
        self.df = df
