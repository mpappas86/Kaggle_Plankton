from net import Net
import numpy as np

class Neural_Net(Net):
    def __init__(self, input_size, output_size):
        super(Neural_Net, self).__init__(input_size, output_size)
        self.cost_function = lambda x,y: 0.5*np.sum(((x-y)*(x-y)).flatten())
        self.dcost = lambda x,y: x-y
        
    def backprop(self, data, target):
        self.input_data(data)
        self.forward_pass()
        net_output = self.retrieve_output()
        self.zero_buffers()
        output_data = self.dcost(net_output, target)
        for output_node, ranges in self.outputs.iteritems():
            output_node.output_buffer[ranges[1],:] = output_data[ranges[0],:]
            output_node.push_downstream()
        for node in self.nodes:
            node.layer.update_weights()

    def cost(self, data, target):
        self.input_data(data)
        self.forward_pass()
        net_output = self.retrieve_output()
        return self.cost_function(net_output, target)
