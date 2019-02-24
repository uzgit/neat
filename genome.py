from copy import deepcopy
from random import *
from functions import *

class NodeGene:

    def __init__(self, identifier, aggregation_function="sum", activation_function="sigmoid", is_input_node=False, is_output_node=False):

        self.identifier = identifier
        self.aggregation_function = aggregation_function
        self.activation_function = activation_function
        self.is_input_node= is_input_node
        self.is_output_node = is_output_node

    def __str__(self):

        representation = "node {}({}): {}, {}".format(self.identifier, self.layer, self.aggregation_function, self.activation_function)

        if self.is_input_node:
            representation += " (input)"
        elif self.is_output_node:
            representation += " (output)"

        return representation

    __repr__ = __str__

    def str(self):

        return self.__str__()

    def __repr__(self):

        return self.__str__()

class EdgeGene:

    def __init__(self, identifier, innovation_number, input_node_identifier, output_node_identifier, weight, is_enabled=True):

        self.identifier = identifier
        self.innovation_number = innovation_number
        self.input_node_identifier = input_node_identifier
        self.output_node_identifier = output_node_identifier
        self.weight = weight
        self.is_enabled = is_enabled

        self.input_node_index = -1
        self.output_node_index = -1

    def __str__(self):

        representation = "edge {} ({}): {}->{}, {}".format(self.identifier, self.innovation_number, self.input_node_identifier, self.output_node_identifier, self.weight)

        if self.is_enabled:
            representation += " enabled"
        else:
            representation += " disabled"

        return representation

    def str(self):

        return self.__str__()

    def __repr__(self):

        return self.__str__()

class Genome:

    def __init__(self, identifier, nodes, edges):

        self.identifier = identifier
        self.nodes = deepcopy(nodes)
        self.nodes.sort(key=lambda node: node.identifier)
        self.edges = deepcopy(edges)

        self.num_inputs  = len([node for node in self.nodes if node.is_input_node])
        self.num_outputs = len([node for node in self.nodes if node.is_output_node])

    def mutate_add_node(self, new_node_identifier, innovation_number_1, innovation_number_2, mode=None, new_node_aggregation_function="sum", new_node_activation_function="sigmoid"):

        active_edges = [edge for edge in self.edges if edge.is_enabled]

        # choose a random active edge
        random_active_edge = choice(active_edges)
        random_active_edge.is_enabled = False

        if mode == "random":
            new_node_aggregation_function = choice(aggregation_function_names)
            new_node_activation_function = choice(activation_function_names)

        # create a new node
        new_node = NodeGene(new_node_identifier, new_node_aggregation_function, new_node_activation_function)

        # create two new edges
        new_edge_1 = EdgeGene(innovation_number_1, innovation_number_1, random_active_edge.input_node_identifier, new_node.identifier, 1)
        new_edge_2 = EdgeGene(innovation_number_2, innovation_number_2, new_node.identifier, random_active_edge.output_node_identifier, random_active_edge.weight)

        self.nodes.append(new_node)
        self.edges.append(new_edge_1)
        self.edges.append(new_edge_2)

    def mutate_remove_node(self):

        # choose a random node to remove
        removed_node = choice([node for node in self.nodes if not node.is_input_node and not node.is_output_node])
        # remove the relevant node
        self.nodes.remove(removed_node)
        print(removed_node.identifier)

        # get all the enabled edges that use the removed node as input or output
        removed_edges = [edge for edge in self.edges if edge.is_enabled and (edge.input_node_identifier == removed_node.identifier or edge.output_node_identifier == removed_node.identifier)]
        # disable the relevant edges
        for edge in removed_edges:
            edge.is_enabled = False

    def mutate_reset_weight(self, initial_weight_min, initial_weight_max):

        reset_edge = choice([edge for edge in self.edges])
        new_weight = uniform(initial_weight_min, initial_weight_max)
        reset_edge.weight = new_weight

    def mutate_scale_weight(self, weight_min, weight_max):

        mutated_edge = choice([edge for edge in self.edges])
        mutated_edge.weight *= uniform(weight_min, weight_max)

    def mutate_change_aggregation_function(self):

        mutated_node = choice([node for node in self.nodes if not node.is_input_node and not node.is_output_node])
        new_aggregation_function = choice(aggregation_function_names)
        mutated_node.aggregation_function = new_aggregation_function

    def mutate_change_activation_function(self):

        mutated_node = choice([node for node in self.nodes if not node.is_input_node and not node.is_output_node])
        new_activation_function = choice(activation_function_names)
        mutated_node.activation_function = new_activation_function

    def __str__(self):

        representation = "Genome {}:\n".format(self.identifier)

        for node in self.nodes:
            representation += "\t" + node.str() + "\n"

        for edge in self.edges:
            representation += "\t" + edge.str() + "\n"

        representation += "\n"

        return representation

    def str(self):

        return self.__str__()

    def __repr__(self):

        return self.__str__()