import numpy as np
from net_node import Net_Node

class Neural_Node(Net_Node):
    def __init__(self, layer, name="Anonymous Node", p=None, latch_step=False):
        super(Neural_Node, self).__init__(layer, name=name, p=p, latch_step=latch_step)
        self.upstream_checkin = 0
        self.dataup = None
        self.savedup = None
    
    def take_upstream(self, upstream, output_node):
        self.output_buffer[self.outputs[output_node],:] += upstream
        self.upstream_checkin = self.upstream_checkin + 1
            
    def push_downstream(self):
        if not self.upstream_checkin == len(self.outputs):
            return
        self.upstream_checkin = 0
        self.input_buffer[self.dropout_in,:] = self.layer.backprop(self.output_buffer, self.dropout_in, self.dropout_array, self)
        for input_node, input_range in self.inputs.iteritems():
            input_node.take_upstream(self.input_buffer[input_range,:], self)
            input_node.push_downstream()
