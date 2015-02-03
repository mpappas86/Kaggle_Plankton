import numpy as np

# A Layer defines a mapping between one vector and another vector of the form output = function(matrix*input + bias).  The layer also keeps learning rates and momentum vectors for each of the weights in the matrix and bias, along with momentum sacle factors and a regularization penalty on the matrix
class Layer(object):
    globalindex = 0
    
    def __init__(self, input_size, output_size, order=None):
        if order == None:
            self.order = Layer.globalindex
            Layer.globalindex = Layer.globalindex+1
        else:
            self.order = order
        self.shape = (output_size, input_size)
        self.W = np.random.normal(0,1,(output_size, input_size))
        self.b = np.random.normal(0,1,(output_size,1))
        self.f = lambda x: x
        self.lr = 0.1
        self.lam = 0
        self.nu = 0
        self.lrb = 0.1
        self.nub = 0
        self.c = None
        self.Wspeed = np.zeros(self.W.shape)
        self.bspeed = np.zeros(self.b.shape)
        self.Wgrad = np.zeros(self.W.shape)
        self.bgrad = np.zeros(self.b.shape)
        self.batch_size = None
        self.updated_yet = True
        
    def predict(self, data, node):
        return self.f(self.W.dot(data)+self.b)
    
    def training_predict(self, data, dropout_in, dropout_array, node):
        ii = np.where(dropout_in)[0]
        oi = np.where(dropout_array)[0][:,np.newaxis]
        return self.f(self.W[oi,ii].dot(data[dropout_in,:])+self.b[dropout_array,:])
    
    def get_num_params(self):
        return np.size(self.W)+np.size(self.b)
    
    def set_weights(self, W, b):
        self.W = W
        self.b = b

    def get_weights(self):
        return self.W, self.b

    # Returns a 1D flattened vector of the weights, in the form [<row 1 of W>, <row 2 of W>,... <row N of W>, <b>]
    def get_weight_vector(self):
        return np.concatenate((self.W.flatten(), self.b.flatten()))

    def set_weight_vector(self, weights):
        self.W = weights[:np.size(self.W)].reshape(self.W.shape)
        self.b = weights[np.size(self.W):].reshape(self.b.shape)

    def update_weights(self):
        if not self.updated_yet:
            self.Wspeed = self.nu*self.Wspeed - (self.lr*self.lam)*self.W - self.lr/self.batch_size*self.Wgrad
            self.bspeed = self.nub*self.bspeed - self.lrb/self.batch_size*self.bgrad
            self.W += self.Wspeed
            self.b += self.bspeed
            if not self.c is None:
                for row in self.W:
                    b = np.linalg.norm(row)
                    if b > self.c:
                        row[:] = row[:]*(c/b)
            self.updated_yet = True
        
    def set_f(self,f):
        self.f = f

    def set_lrates(self, lr, lrb):
        self.lr = lr
        self.lrb = lrb

    def set_momenta(self, nu, nub):
        self.nu = nu
        self.nub = nub

    def set_weight_decay(self, lam):
        self.lam = lam

    def set_row_norm(self, c):
        if c is None:
            self.c = None
        else:
            self.c = float(c)

    def set_hyper_params(self, lr=0.1, lrb=0.1, nu=0.0, nub=0.0, lam=0.0, c=None):
        self.lr = lr
        self.lrb = lrb        
        self.nu = nu
        self.nub = nub
        self.lam = lam        
        self.c = c
        
