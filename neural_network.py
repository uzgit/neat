import os
import sys

# For use in contexts where this file is imported from outside this directory.
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

from genome import *

class Node:

    def __init__(self, node_gene):

        assert isinstance(node_gene, NodeGene)

        # Copy attributes from node_gene.
        self.identifier = node_gene.identifier
        self.aggregation_function = node_gene.aggregation_function
        self.bias = node_gene.bias
        self.activation_function = node_gene.activation_function
        self.is_input_node  = node_gene.is_input_node
        self.is_output_node = node_gene.is_output_node
        self.is_enabled = node_gene.is_enabled

        # Initialize own attributes.
        self.layer       = None
        self.inputs      = None
        self.aggregation = None
        self.activation  = None

        # This is of the form [[Node, weight], [Node, weight], ...]
        self.outputs = []

    def add_input(self, input_value):

        if self.inputs is None:
            self.inputs = []

        self.inputs.append(input_value)

    def aggregate(self):

        if self.bias is not None:
            self.add_input(self.bias)

        assert self.inputs is not None
        assert None not in self.inputs

        self.aggregation = self.aggregation_function(self.inputs)

    def activate(self):

        assert self.aggregation is not None
        self.activation = self.activation_function(self.aggregation)

    def propagate(self):

        assert self.activation is not None

        for output_node, weight in self.outputs:
            output_node.add_input(self.activation * weight)

    def clear(self):

        self.inputs      = None
        self.aggregation = None
        self.activation  = None

    def __str__(self):

        representation = "Node {} (layer {})".format(self.identifier, self.layer)
        representation += ", aggregation: {}, bias: {}".format(function_names[self.aggregation_function], self.bias)
        representation += ", activation: {}".format(function_names[self.activation_function])

        representation += ", outputs: "
        for output_node, weight in self.outputs:
            representation += "[node {}, weight {}]".format(output_node.identifier, round(weight, 2))

        if not self.is_enabled:
            representation += " (disabled)"

        return representation

class Edge:

    def __init__(self, edge_gene):

        assert isinstance(edge_gene, EdgeGene)

        self.identifier = edge_gene.identifier
        self.innovation_number = edge_gene.innovation_number
        self.weight     = edge_gene.weight
        self.input_node_identifier  = edge_gene.input_node_identifier
        self.output_node_identifier = edge_gene.output_node_identifier
        self.is_enabled = edge_gene.is_enabled

        # Edge objects get a layer which input_node.layer + 0.5
        self.layer = None
        self.input_node  = None
        self.output_node = None

    def __str__(self):

        representation = "edge {}->{}, {}".format(self.input_node.identifier, self.output_node.identifier, round(self.weight, 2))
        if not self.is_enabled:
            representation += " (disabled)"
        return representation

class FeedForwardNeuralNetwork:

    def __init__(self, genome):

        self.genome = genome
        self.identifier  = genome.identifier
        self.num_inputs  = genome.num_inputs
        self.num_outputs = genome.num_outputs

        self.nodes = None
        self.edges = None

        self.generate_nodes()
        self.generate_edges()

        self.nodes.sort(key=lambda node: node.identifier)
        self.input_nodes  = [node for node in self.nodes if node.is_input_node]
        self.hidden_nodes = [node for node in self.nodes if not node.is_input_node and not node.is_output_node]
        self.output_nodes = [node for node in self.nodes if node.is_output_node]

        self.num_hidden_nodes = len([node for node in self.nodes if not node.is_input_node and not node.is_output_node and node.is_enabled])

        self.set_topology()

        self.active_nodes = [node for node in self.nodes if node.layer is not None]
        self.active_nodes.sort(key = lambda node: node.layer)

        self.min_layer = min(node.layer for node in self.nodes if node.layer is not None)
        self.max_layer = max(node.layer for node in self.nodes if node.layer is not None)

    def generate_nodes(self):

        self.nodes = []
        for node_gene in self.genome.nodes:

            node = Node(node_gene)
            self.nodes.append( node )

    def get_node(self, identifier):

        node = ([node for node in self.nodes if node.identifier == identifier] + [None])[0]
        return node

    def generate_edges(self):

        self.edges = []
        for edge_gene in self.genome.edges:

            edge = Edge( edge_gene )
            edge.input_node  = self.get_node(edge.input_node_identifier)
            edge.output_node = self.get_node(edge.output_node_identifier)

            assert edge.input_node  is not None, "input node {} is apparently None".format(edge.input_node_identifier)
            assert edge.output_node is not None, "output_node {} is apparently None".format(edge.output_node_identifier)
            if edge.is_enabled:
                assert edge.input_node.is_enabled is True
                assert edge.output_node.is_enabled is True

            self.edges.append( edge )

    def set_topology(self):

        for current_node in self.nodes:
            for edge in self.edges:

                if edge.input_node == current_node and edge.is_enabled:

                    current_node.outputs.append([edge.output_node, edge.weight])

        # Iterate from input nodes to output nodes, setting layers and output nodes for each node.
        current_layer = 1
        node_stack = [node for node in self.input_nodes]
        while len(node_stack) > 0:

            new_node_stack = []

            for current_node in node_stack:

                current_node.layer = current_layer

                for edge in self.edges:
                    if edge.input_node_identifier == current_node.identifier:
                        edge.layer = current_layer + 0.5

                # current_node.outputs = [[edge.output_node, edge.weight] for edge in self.edges if edge.input_node_identifier == current_node.identifier and edge.is_enabled]

                new_node_stack += [edge.output_node for edge in self.edges if edge.input_node_identifier == current_node.identifier and edge.is_enabled]

            node_stack = [] + new_node_stack
            current_layer += 1

        change_occurred = True
        while change_occurred:
            change_occurred = False

            for current_node in self.nodes:

                if current_node.layer == None and len(current_node.outputs) > 0:

                    output_layers = [output[0].layer for output in current_node.outputs if output[0].layer is not None]

                    if len(output_layers) > 0:
                        current_node.layer = min(output_layers) - 1
                        change_occurred = True

    def activate(self, inputs):

        assert len(inputs) == self.num_inputs

        for node in self.nodes:
            node.clear()

        # Set inputs.
        for i in range(self.num_inputs):
            self.input_nodes[i].add_input(inputs[i])

        # Propagate in order. Nodes are sorted by layer.
        for node in self.active_nodes:
            node.aggregate()
            node.activate()
            node.propagate()

        return [node.activation for node in self.output_nodes]

    def __str__(self):

        representation = "Network {}: {}->{}->{}".format(self.identifier, self.num_inputs, self.num_hidden_nodes, self.num_outputs)

        representation += "\n\tNodes:"
        sorted_nodes = sorted(self.nodes, key=lambda node : node.identifier)
        for node in sorted_nodes:
            representation += "\n\t\t" + str(node)

        representation += "\n\tEdges:"
        for edge in self.edges:
            representation += "\n\t\t" + str(edge)

        return representation
