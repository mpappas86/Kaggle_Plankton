from net import Net
import numpy as np

class Neural_Net(Net):
    def __init__(self, input_size, output_size):
        super(Neural_Net, self).__init__(input_size, output_size)
        self.cost_function = lambda x,y: 0.5*np.sum(((x-y)*(x-y)).flatten())
        self.dcost = lambda x,y: x-y

    def make_mini(self, data, startindex, mbsize):
        newindex = startindex+mbsize
        if newindex <= data.shape[1]:
            return data[:,startindex:newindex], newindex
        else:
            newindex = newindex - data.shape[1]
        return np.concatenate((data[:,startindex:], data[:,:newindex]),1), newindex

    def train_mini(self, data, labels, mbsize, epochs, tag='', taginc=100):
        self.set_buffer_depth(mbsize)
        rcerror = []
        index = 0
        lim = data.shape[1]*epochs/float(mbsize)
        for y in xrange(int(lim)):
            mbdata, phony = self.make_mini(data, index, mbsize)
            mblabels, index = self.make_mini(labels, index, mbsize)
            self.backprop(mbdata, mblabels)
            rcerror.append(self.cost(mbdata,mblabels))
            if (y % taginc) == taginc-1:
                print tag+str(y),100*round(y/lim,4),sum(rcerror[-taginc:])/float(taginc)
        return rcerror
    
    def backprop(self, data, target):
        self.input_data(data)
        self.forward_pass()
        net_output = self.retrieve_output()
        self.zero_buffers()
        output_data = self.dcost(net_output, target)
        for node in self.nodes:
            node.layer.backprop_setup()
        for output_node, ranges in self.outputs.iteritems():
            output_node.output_buffer[ranges[1],:] += output_data[ranges[0],:]
            output_node.push_downstream()
        for node in self.nodes:
            node.layer.update_weights()

    def cost(self, data, target):
        self.input_data(data)
        self.forward_pass()
        net_output = self.retrieve_output()
        return self.cost_function(net_output, target)
