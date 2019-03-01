from genome import *

# for visualization purposes only
class Edge:

    def __init__(self, input_node_identifier, output_node_identifier, weight):
        self.input_node_identifier = input_node_identifier
        self.output_node_identifier = output_node_identifier
        self.weight = weight

    def __str__(self):
        return "edge {}->{}, {}".format(self.input_node_identifier, self.output_node_identifier, self.weight)

    def str(self):
        return self.__str__()

class Node:

    def __init__(self, identifier, aggregation_function, activation_function, layer=-1, outputs=None, is_input_node=False, is_output_node=False, bias=None):

        self.identifier = identifier
        self.aggregation_function = aggregation_function
        self.activation_function = activation_function

        self.bias = bias

        self.layer = layer

        if outputs == None:
            self.outputs = []
        else:
            self.outputs = deepcopy(outputs)

        self.is_input_node = is_input_node
        self.is_output_node = is_output_node

        self.inputs = None
        self.aggregation = None
        self.activation  = None

    def add_input(self, input_value):

        if self.inputs is None:
            self.inputs = []

        self.inputs.append(input_value)

    def aggregate(self):
        assert None not in self.inputs

        if self.bias is not None:
            self.add_input(self.bias)

        self.aggregation = aggregation_functions[self.aggregation_function](self.inputs)
        return self.aggregation

    def activate(self):

        assert self.aggregation is not None
        self.activation = activation_functions[self.activation_function](self.aggregation)
        return self.activation

    def clear(self):

        if self.inputs is not None:
            self.inputs.clear()
        self.aggregation = None
        self.activation  = None

    def propagate(self):

        assert self.activation is not None
        for node, weight in self.outputs:
            node.add_input(self.activation * weight)

    def __str__(self):

        representation = "Node {} (layer {}), aggregation: {}, bias: {}, activation: {}, output links: ".format(self.identifier, self.layer, self.aggregation_function, self.bias, self.activation_function)

        if len(self.outputs) == 0:
            representation += "[]"
        else:
            for node, weight in self.outputs:
                representation += "[node {} : weight {}]".format(node.identifier, weight)

        return representation


class FeedForwardNeuralNetwork:

    def __init__(self, genome):

        ################################################################
        #self.genome = deepcopy(genome)
        self.genome = genome
        self.num_inputs = genome.num_inputs
        self.num_outputs = genome.num_outputs
        self.identifier = genome.identifier

        self.nodes = []
        self.edges = []

        self.generate_nodes()


        # sort the nodes by identifier first, for display purposes
        self.nodes.sort(key=lambda node: node.identifier)

        # # if no nodes are linked to the output nodes:
        # if len([node for node in self.nodes if node.is_output_node]) == 0:
        #     for node in [node for node in self.genome.nodes if node.is_output_node]:
        #         self.nodes.append(Node(node.identifier, node.aggregation_function, node.activation_function, layer=2, is_output_node=True))

        self.input_nodes  = [node for node in self.nodes if node.is_input_node]
        self.hidden_nodes = [node for node in self.nodes if not node.is_input_node and not node.is_output_node]
        self.output_nodes = [node for node in self.nodes if node.is_output_node]

        # sort the nodes by layer for activation purposes
        self.nodes.sort(key=lambda node: node.layer)

        self.num_layers = max([node.layer for node in self.nodes])
        self.num_hidden_nodes = len(self.hidden_nodes)

    def generate_nodes(self):

        node_stack = []

        # generate input nodes
        for input_node_gene in [node_gene for node_gene in self.genome.nodes if node_gene.is_input_node]:
            new_input_node = Node(input_node_gene.identifier, aggregation_function=input_node_gene.aggregation_function, bias=input_node_gene.bias, activation_function=input_node_gene.activation_function)
            new_input_node.is_input_node = True
            node_stack.append(new_input_node)

            self.nodes.append(new_input_node)

        # generate output_nodes
        processed_nodes = []
        layer_number = 1
        while(len(node_stack) > 0):

            next_node_stack = []
            for current_node in node_stack:

                current_node.layer = layer_number

                current_node_edges = [edge for edge in self.genome.edges if edge.input_node_identifier == current_node.identifier and edge.is_enabled]

                for current_edge in current_node_edges:

                    next_node_gene = [node for node in self.genome.nodes if node.identifier == current_edge.output_node_identifier][0]

                    #next_node = None
                    if next_node_gene.identifier in [node.identifier for node in self.nodes]:
                        next_node = [node for node in self.nodes if node.identifier == next_node_gene.identifier][0]
                    else:
                        next_node = Node(next_node_gene.identifier, aggregation_function=next_node_gene.aggregation_function, bias=next_node_gene.bias, activation_function=next_node_gene.activation_function, is_output_node=next_node_gene.is_output_node)
                        self.nodes.append(next_node)

                    if next_node not in next_node_stack:
                        next_node_stack.append(next_node)

                    # don't set a node's outputs more than once
                    if current_node not in processed_nodes:
                        current_node.outputs.append([next_node, current_edge.weight])
                        self.edges.append(Edge(current_node.identifier, next_node.identifier, current_edge.weight))

            # replace node_stack with next_node_stack
            processed_nodes += node_stack
            node_stack = [] + next_node_stack

            layer_number += 1

    def activate(self, inputs):

        assert len(inputs) == self.num_inputs, "Expected {} inputs but got {} inputs".format(self.num_inputs, len(inputs))

        # clear all data
        for node in self.nodes:
            node.clear()

        # propagate inputs first
        for i in range(self.num_inputs):
            self.nodes[i].add_input(inputs[i])

        # propagate
        for node in self.nodes:
            node.aggregate()
            node.activate()
            node.propagate()

        # return output node activations in order of node identifier
        return [node.activation for node in self.output_nodes]

    def __str__(self):

        representation = "neural network ({} layers) from genome {}".format(self.num_layers, self.identifier)
        representation += "\n\t{}->{}->{}".format(self.num_inputs, self.num_hidden_nodes, self.num_outputs)

        representation += "\n\tInput node(s):"
        for node in self.input_nodes:
            representation += "\n\t\t" + str(node)

        if self.num_hidden_nodes > 0:
            representation += "\n\tHidden node(s):"
            for node in sorted(self.hidden_nodes, key=lambda node: node.layer):
                representation += "\n\t\t" + str(node)

        representation += "\n\tOutput node(s):"
        for node in sorted(self.output_nodes, key=lambda node: node.layer):
            representation += "\n\t\t" + str(node)

        # for edge in self.edges:
        #     representation += "\n\t{}".format(str(edge))

        return representation

    def num_nodes(self):

        return len(self.nodes)

    def num_edges(self):

        return len(self.edges)

    def str(self):

        return self.__str__()

    def __repr__(self):

        return self.__str__()