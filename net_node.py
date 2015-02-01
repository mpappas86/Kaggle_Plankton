import numpy as np

class DiagnoseError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

class Net_Node(object):
    def __init__(self, layer, name="Anonymous Node", p=None, latch_step = False):
        self.layer = layer
        self.name = name
        self.input_buffer = np.zeros((self.layer.shape[1],1))
        self.true_input = np.zeros((self.layer.shape[1],1))
        self.output_buffer = np.zeros((self.layer.shape[0],1))
        self.dropout_array = np.ones(self.layer.shape[1],dtype=bool)
        self.dropout_in = np.ones(self.layer.shape[1],dtype=bool)        
        self.inputs = {}
        self.outputs = {}
        self.p = p
        self.latch_step = latch_step
        self.input_checkin = 0

    def setp(self, p):
        self.p = p

    def get_num_params(self, *args, **kwargs):
        return self.layer.get_num_params()

    def set_weights(self, *args, **kwargs):
        self.layer.set_weights(*args, **kwargs)

    def get_weights(self, *args, **kwargs):
        return self.layer.get_weights(*args, **kwargs)

    def set_weight_vector(self, *args, **kwargs):
        self.layer.set_weight_vector(*args, **kwargs)

    def get_weight_vector(self, *args, **kwargs):
        return self.layer.get_weight_vector(*args, **kwargs)

    def zero_buffers(self):
        self.input_buffer = np.zeros(self.input_buffer.shape)
        self.output_buffer = np.zeros(self.output_buffer.shape)

    def take_input(self, data, input_node):
        self.input_buffer[self.inputs[input_node],:] = data
        self.input_checkin = self.input_checkin + 1

    def training_take_input(self, data, dropout, input_node):
        self.input_buffer[self.inputs[input_node],:] = data
        self.dropout_in[self.inputs[input_node]] = dropout
        self.input_checkin = self.input_checkin + 1

    # TODO: Make this work with dropout?  Does that make sense?
    def latch_input(self):
        self.true_input = self.input_buffer.copy()

    def push_output(self):
        # If all of the inputs have checked in and pushed their input to this node's buffer, then continue
        if not self.input_checkin == len(self.inputs):
            return
        # reset for next push
        self.input_checkin = 0
        # If we want the input to flow through the node in one step, go ahead and latch the input buffer to the true input before running the calculation
        if not self.latch_step:
            self.latch_input()
        # Calculate the output of this node
        self.output_buffer = self.layer.predict(self.true_input, self)
        # Push the output to all of the nodes depending on this node for input
        for output_node, output_range in self.outputs.iteritems():
            output_node.take_input(self.output_buffer[output_range,:], self)
            # If this node was the last input to check in, the output layer will now push its output
            output_node.push_output()

    def traiing_push_output(self):
        # If all of the inputs have checked in and pushed their input to this node's buffer, then continue
        if not self.input_checkin == len(self.inputs):
            return
        # reset for next push
        self.input_checkin = 0
        # If we want the input to flow through the node in one step, go ahead and latch the input buffer to the true input before running the calculation
        if not self.latch_step:
            self.latch_input()
        if p is not None:
            self.dropout_array = np.random.rand(self.layer.shape[1]) < self.p
        else:
            self.dropout_array = np.ones(self.layer.shape[1],dtype=bool)
        # Calculate the output of this node
        self.output_buffer = self.layer.trainig_predict(self.true_input, self.dropout_in, self.dropout_array, self)
        # Push the output to all of the nodes depending on this node for input
        for output_node, output_range in self.outputs.iteritems():
            output_node.trainig_take_input(self.output_buffer[output_range,:], self.dropout_arry[output_range], self)
            # If this node was the last input to check in, the output layer will now push its output
            output_node.push_output()
            
    def set_buffer_depth(self, depth):
        self.input_buffer = np.zeros((self.layer.shape[1],depth))
        self.true_input = np.zeros((self.layer.shape[1],depth))
        self.output_buffer = np.zeros((self.layer.shape[0],depth))

    def add_input(self, input_node, input_range):
        self.inputs[input_node] = input_range

    def add_output(self, output_node, output_range):
        self.outputs[output_node] = output_range

    def remove_input(self, input_node):
        return self.inputs.pop(input_node, None)

    def remove_output(self, output_node):
        return self.outputs.pop(output_node, None)

    def diagnose(self):
        # Check that all inputs have this node as an output, and that the ranges are the same size
        for input_node, input_range in self.inputs.iteritems():
            if not self in input_node.outputs:
                raise DiagnoseError(str(self) + " has " + str(input_node) + " as an input, but is not reciprocated as an output")
            elif not len(input_range) == len(input_node.outputs[self]):
                raise DiagnoseError(str(self) + " has " + str(input_node) + " as an input, but their ranges do not agree")
        # Check that all input nodes are accounted for, and that no input nodes overlap
        input_set_length = len(reduce(set.union,imap(set,self.inputs.values())))
        if not input_set_length == sum([len(x) for x in self.inputs.values()]):
            raise DiagnoseError(str(self) + " has overlapping inputs")
        if not input_set_length == self.input_buffer.shape[0]:
            raise DiagnoseError(str(self) + " has some inputs unaccounted for")

    def set_layer(self, layer):
        self.layer = layer

    def set_inputs(self, inputs):
        self.inputs = inputs

    def set_outputs(self, outputs):
        self.outputs = outputs

    def __repr__(self):
        return "<" + self.name + " " + repr(self.layer.shape) + " (" + repr(id(self)) + ")>"
