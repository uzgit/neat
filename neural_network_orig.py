from genome import *

class NeuralNetwork:

    def __init__(self, genome):

        ################################################################
        # don't copy the genome members - just reference them
        self.nodes = genome.nodes
        self.edges = genome.edges

        self.input_nodes  = [node for node in self.nodes if node.is_input_node]
        self.output_nodes = [node for node in self.nodes if node.is_output_node]
        self.hidden_nodes = [node for node in self.nodes if not (node.is_input_node or node.is_output_node)]

        self.input_nodes.sort(key=lambda node: node.identifier)
        self.hidden_nodes.sort(key=lambda node: node.identifier)
        self.output_nodes.sort(key=lambda node: node.identifier)

        self.genome = genome
        self.identifier = genome.identifier
        self.num_inputs  = len(self.input_nodes)
        self.num_outputs = len(self.output_nodes)
        ################################################################

        # generate phenotype
        self.initialize_members()
        self.determine_layers()
        self.nodes.sort(key=lambda node: node.layer)

        self.active_nodes = [node for node in self.nodes if node.layer != -1]
        self.active_nodes.sort(key=lambda node: node.identifier)
        self.active_nodes.sort(key=lambda node: node.layer)

        # set indices of active nodes
        for i in range(len(self.active_nodes)):
            self.active_nodes[i].index = i

        # propagate node indice to edges
        for current_node in self.active_nodes:
            for edge in self.edges:
                if edge.input_node_identifier == current_node.identifier:
                    edge.input_node_index = current_node.index
                if edge.output_node_identifier == current_node.identifier:
                    edge.output_node_index = current_node.index
        ################################################################

    def initialize_members(self):

        for node in self.nodes:
            node.layer = -1
            node.index = -1
            node.aggregation = 0
            node.activation = 0

        for edge in self.edges:
            edge.input_node_index = -1
            edge.output_node_index = -1

    def determine_layers(self):

        node_stack = []
        for input_node in self.input_nodes:
            node_stack.append(input_node)

        current_layer = 0
        while len(node_stack) > 0:

            current_layer += 1

            edge_stack = []
            for current_node in node_stack:
                current_node.layer = current_layer
                edge_stack += [edge for edge in self.edges if edge.input_node_identifier == current_node.identifier and edge.is_enabled]

            node_stack = []
            for edge in edge_stack:
                node_stack += [node for node in self.nodes if edge.output_node_identifier == node.identifier]

        self.sanity_check()

        self.num_layers = max([node.layer for node in self.output_nodes])

        return self.num_layers

    def sanity_check(self):

        node_identifiers = [node.identifier for node in self.nodes]

        for edge in self.edges:
            assert edge.input_node_identifier in node_identifiers, "input node {} of edge {} is not in self.nodes".format(
                edge.input_node_identifier, edge.identifier)
            assert edge.output_node_identifier in node_identifiers, "output node {} of edge {} is not in self.nodes".format(
                edge.output_node_identifier, edge.identifier)

    def activate(self, inputs):

        assert len(inputs) == self.num_inputs, "Expected {} inputs but got {} inputs".format(self.num_inputs, len(inputs))

        for node in self.active_nodes:
            node.aggregation = 0
            node.activation = 0

        for i in range(len(self.input_nodes)):
            self.input_nodes[i].aggregation = inputs[i]

        # process input nodes
        # easier to process all input nodes, not just active ones, because some input nodes can be inactive
        node_stack = [node for node in self.input_nodes]
        edge_stack = []

        for current_node in self.input_nodes:
            # input nodes have a single input. aggregations have already been set. activate using the specified actiation function
            current_node.activation = activation_functions[current_node.activation_function](current_node.aggregation)

            print(current_node.activation)

            # aggregate to downstream nodes
            current_node_edges = [edge for edge in self.edges if edge.input_node_identifier == current_node.identifier]
            #for edge in current_node_edges:
            #    self.nodes[edge.output_node_index].aggregation += current_node.activation

            # add the current nodes edges to the stack
            edge_stack += current_node_edges

        # process each layer
        for current_layer in range(2, self.num_layers + 1):

            # get the nodes in the current layer
            current_layer_nodes = [node for node in self.active_nodes if node.layer == current_layer]

            # aggregate
            for current_node in current_layer_nodes:

                print("aggregating for node", current_node.identifier, "of layer", current_node.layer)

                current_node_input_edges = [edge for edge in self.edges if edge.output_node_identifier == current_node.identifier]
                print("\t", current_node_input_edges)

                for input_edge in current_node_input_edges:
                    print(input_edge.input_node_identifier, input_edge.input_node_index, "->", input_edge.output_node_identifier, input_edge.output_node_index)

                    print(self.nodes[input_edge.input_node_index].aggregation)
                    current_node.aggregation += self.nodes[input_edge.input_node_index].activation

                print(current_node.aggregation)

                current_node.activation = activation_functions[current_node.activation_function](current_node.aggregation)

        outputs = [node.activation for node in self.output_nodes]

        return outputs

    def __str__(self):

        representation = "neural network ({} layers) from genome {}".format(self.num_layers, self.identifier)
        representation += "\n\t {}->{}->{}".format(self.num_inputs, len(self.hidden_nodes), self.num_outputs)

        for node in sorted(self.nodes, key = lambda node: node.layer):
            representation += "\n\t" + node.str()
            for edge in sorted([edge for edge in self.edges if edge.input_node_identifier == node.identifier], key=lambda edge: edge.output_node_identifier):
                representation += "\n\t\t" + edge.str()

        return representation

    def str(self):

        return self.__str__()

    def __repr__(self):

        return self.__str__()