import numpy as np

# connections is list with elements of the form (<input_node, output_node, number of connections>).  If a node should connect to the input of the net, use "input" in place of input_node.  If a node should connect to the output of the net, use "output" in place of output_node.
# we don't currently deal with wraparound
def autoconnect(net, connections, verbose=True):
    # pairs of the form [input counter, output counter]
    connectionlist = {}
    connectionlist["input"] = [0,0]
    connectionlist["output"] = [0,0]
    for connection in connections:
        (input_node, output_node, num_connections) = connection
        if not input_node in connectionlist:
            connectionlist[input_node] = [0,0]
        if not output_node in connectionlist:
            connectionlist[output_node] = [0,0]
        inpval = connectionlist[input_node][1]
        inpend = inpval + num_connections
        outval = connectionlist[output_node][0]
        outend = outval + num_connections
        if input_node == "input":
            net.add_input(output_node, (xrange(inpval,inpend), xrange(outval,outend)))
        elif output_node == "output":
            net.add_output(input_node, (xrange(outval,outend), xrange(inpval,inpend)))
        else:
            net.add_node(input_node)
            net.add_node(output_node)
            output_node.add_input(input_node, xrange(outval,outend))
            input_node.add_output(output_node, xrange(inpval,inpend))
        connectionlist[input_node][1] = inpend
        connectionlist[output_node][0] = outend
