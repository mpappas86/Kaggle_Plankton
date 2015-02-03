from net import Net
import numpy as np

class Neural_Net(Net):
    def __init__(self, input_size, output_size, p=None, cost_function=None, dcost=None):
        super(Neural_Net, self).__init__(input_size, output_size, p=p)
        assert((cost_function is None)  == (dcost is None))
        if(cost_function is None):
            self.cost_function = lambda x,y: 0.5*((x-y)*(x-y)).sum(0)
            self.dcost = lambda x,y: x-y
        else:
            self.cost_function = cost_function
            self.dcost = dcost

    def make_mini(self, data, startindex, mbsize):
        newindex = startindex+mbsize
        if newindex <= data.shape[1]:
            return data[:,startindex:newindex], newindex
        else:
            newindex = newindex - data.shape[1]
        return np.concatenate((data[:,startindex:], data[:,:newindex]),1), newindex

    def train_mini(self, data, labels, mbsize, epochs, tag='', taginc=100, valid_data=None, valid_labels=None):
        self.set_buffer_depth(mbsize)
        rcerror = []
        index = 0
        verror = []
        vindex = 0
        validation = (valid_data is not None) and (valid_labels is not None)
        lim = data.shape[1]*epochs/float(mbsize)
        for y in xrange(int(lim)):
            mbdata, phony = self.make_mini(data, index, mbsize)
            mblabels, index = self.make_mini(labels, index, mbsize)
            self.backprop(mbdata, mblabels)
            rcerror.append(self.cost(mbdata,mblabels))
            if (y % taginc) == taginc-1:
                if not validation:
                    print tag+str(y),100*round(y/lim,4),sum(rcerror[-taginc:])/(float(taginc)*float(mbsize))
                else:
                    vdata, phony = self.make_mini(valid_data, vindex, mbsize)
                    vlabels, vindex = self.make_mini(valid_labels, vindex, mbsize)
                    verror.append(self.cost(vdata, vlabels))
                    print tag+str(y),100*round(y/lim,4),sum(rcerror[-taginc:])/(float(taginc)*float(mbsize)), verror[-1]/float(mbsize)
        return rcerror, verror
    
    def backprop(self, data, target):
        self.training_input_data(data)
        self.training_forward_pass()
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
        return self.cost_function(net_output, target).sum()

    def check_cost(self, data, target):
        self.input_data(data)
        self.check_forward_pass()
        net_output = self.retrieve_output()
        return self.cost_function(net_output, target).sum()
