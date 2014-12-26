import numpy as np
from net_node import Net_Node, DiagnoseError

class Net(object):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.nodes = []
        # inputs maps <input node> to (range of input data, range of input node input)
        self.inputs = {}
        # outputs maps <output node> to (range of output data,range of output node output)
        self.outputs = {}
        self.depth = 0

    # self.inputs is a mapping from <input_nodes> to <range of the input data vector that they should take as inputs>.  Unlike all other nodes, input nodes take all of their input from one source - the user.  This function puts the correct portion of the user's data into each input node
    def input_data(self, data):
        for input_node, ranges in self.inputs.iteritems():
            input_node.input_buffer[ranges[1],:] = data[ranges[0], :]

    def retrieve_output(self):
        output_data = np.zeros((self.output_size,self.depth))
        for output_node, ranges in self.outputs.iteritems():
            output_data[ranges[0],:] = output_node.output_buffer[ranges[1],:]
        return output_data

    def get_layerset(self):
        layers = set()
        for node in self.nodes:
            layers.add(node.layer)
        return layers
    
    def get_weight_vector(self):
        weights = []
        lset = self.get_layerset()
        for layer in lset:
            tmp = layer.get_weight_vector()
            if not tmp is None:
                weights.append(layer.get_weight_vector())
        return np.concatenate(weights)

    def set_weight_vector(self, weights):
        counter = 0
        lset = self.get_layerset()
        for layer in lset:
            num_params = layer.get_num_params()
            if not num_params is None:
                layer.set_weight_vector(weights[counter:(counter+num_params)])
                counter = counter + num_params

    def zero_buffers(self):
        for node in self.nodes:
            node.zero_buffers()
            
    def forward_pass(self):
        for node in self.inputs.keys():
            node.push_output()
        for node in self.nodes:
            if node.latch_step:
                node.latch_input()

    def set_buffer_depth(self, depth):
        self.depth = depth
        for node in self.nodes:
            node.set_buffer_depth(depth)

    def diagnose(self):
        input_set_length = len(reduce(set.union,imap(set,self.inputs.values())))
        if not input_set_length == sum([len(x) for x in self.inputs.values()]):
            raise DiagnoseError("Base Net has overlapping inputs")
        if not input_set_length == self.input_buffer.shape[0]:
            raise DiagnoseError("Base Net has some inputs unaccounted for")
        for node in self.nodes:
            node.diagnose()

    def remove_input(self, input_node):
        self.inputs.pop(input_node, None)
        self.remove_node(input_node)
            
    def add_input(self, input_node, ranges):
        self.inputs[input_node] = ranges
        if not input_node in self.nodes:
            self.add_node(input_node)
            
    def set_input(self, inputs):
        self.inputs = inputs

    def remove_output(self, output_node):
        self.outputs.pop(output_node, None)
        self.remove_node(output_node)
            
    def add_output(self, output_node, ranges):
        self.outputs[output_node] = ranges
        if not output_node in self.nodes:
            self.add_node(output_node)
            
    def set_output(self, outputs):
        self.outputs = outputs

    def add_node(self, node):
        if not node in self.nodes:
            self.nodes.append(node)

    def remove_node(self, node):
        self.nodes.remove(node)
