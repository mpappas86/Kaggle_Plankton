import numpy as np
from net_node import Net_Node, DiagnoseError

class Net(object):
    def __init__(self, input_size, output_size, p=None):
        self.input_size = input_size
        self.output_size = output_size
        self.nodes = []
        # inputs maps <input node> to (range of input data, range of input node input)
        self.inputs = {}
        #outputs maps <output node> to (range of output data,range of output node output)
        self.outputs = {}
        self.depth = 0
        self.p = p
        self.dropout_array = np.ones(self.input_size,dtype=bool)

    # self.inputs is a mapping from <input_nodes> to <range of the input data vector that they should take as inputs>.  Unlike all other nodes, input nodes take all of their input from one source - the user.  This function puts the correct portion of the user's data into each input node
    def input_data(self, data):
        for input_node, ranges in self.inputs.iteritems():
            input_node.input_buffer[ranges[1],:] = data[ranges[0], :]

    def training_input_data(self, data):
        if self.p is not None:
            self.dropout_array = np.random.rand(self.input_size) < self.p
        else:
            self.dropout_array = np.ones(self.input_size,dtype=bool)
        for input_node, ranges in self.inputs.iteritems():
            input_node.input_buffer[ranges[1],:] = data[ranges[0], :]
            input_node.dropout_in[ranges[1]] = self.dropout_array[ranges[0]]

    def retrieve_output(self):
        output_data = np.zeros((self.output_size,self.depth))
        for output_node, ranges in self.outputs.iteritems():
            output_data[ranges[0],:] = output_node.output_buffer[ranges[1],:]
        return output_data

    # def global_setp(self, p=0):
    #     if p is not None and p <= 0:
    #         p = self.p
    #     for node in self.nodes:
    #         node.setp(p)
    #     for node in self.outputs.keys():
    #         node.setp(None)

    def get_ordered_layerset(self):
        layers = set()
        for node in self.nodes:
            try:
                if node.layer.order is not None:
                    layers.add(node.layer)
            except:
                continue
        return sorted(list(layers), key=lambda x: x.order)

    def get_weight_vector(self):
        weights = []
        lset = self.get_ordered_layerset()
        for layer in lset:
            tmp = layer.get_weight_vector()
            if not tmp is None:
                weights.append(layer.get_weight_vector())
        return np.concatenate(weights)

    def set_weight_vector(self, weights):
        counter = 0
        lset = self.get_ordered_layerset()
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

    def training_forward_pass(self):
        for node in self.inputs.keys():
            node.training_push_output()
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

    # ranges is in the form of (<range in the input data, range in the input node>)
    def add_input(self, input_node, ranges):
        self.inputs[input_node] = ranges
        if not input_node in self.nodes:
            self.add_node(input_node)

    def set_input(self, inputs):
        self.inputs = inputs

    def remove_output(self, output_node):
        self.outputs.pop(output_node, None)
        self.remove_node(output_node)

    # ranges is in the form of (<range in the output data, range in the output node>)
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
