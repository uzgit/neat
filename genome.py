from copy import deepcopy
from random import *
import pickle

from functions import *
from globals import *

mutations = {
    "add node" : mutate_add_node_probability,
    "remove node" : mutate_remove_node_probability,
    "set bias" : mutate_set_bias_probability,
    "add edge" : mutate_add_edge_probability,
    "remove edge" : mutate_remove_edge_probability,
    # "reset weight" : mutate_reset_weight_probability,
    "scale weight" : mutate_scale_weight_probability,
     "change aggregation function" : mutate_change_aggregation_function_probability,
     "change activation function" : mutate_change_activation_function_probability,
}

class NodeGene:

    def __init__(self, identifier, aggregation_function=default_aggregation_function, bias=None, activation_function=default_activation_function, is_input_node=False, is_output_node=False, is_enabled=True):

        self.identifier = identifier
        self.aggregation_function = aggregation_function
        self.activation_function = activation_function
        self.is_input_node= is_input_node
        self.is_output_node = is_output_node

        if self.is_input_node or self.is_output_node:
            self.bias = 0
        elif bias == None and random_initial_bias:
            self.bias = uniform(initial_weight_min, initial_weight_max)
        else:
            self.bias = bias

        self.is_enabled = is_enabled

        self.predecessors = []
        self.output_nodes = []
        self.successors = []
        self.layer = None

    def __str__(self):

        representation = "node {}: {}, bias: {}, {}".format(self.identifier, self.aggregation_function, self.bias, self.activation_function)

        if not self.is_enabled:
            representation += " (disabled) "
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

        self.sanity_check()

    def is_equivalent_to(self, edge_2):

        result = False
        if self.input_node_identifier == edge_2.input_node_identifier and self.output_node_identifier == edge_2.output_node_identifier and self.is_enabled == edge_2.is_enabled:
            result = True

        return result

    def sanity_check(self):

        assert self.weight >= global_weight_min, "edge weight {} < {}".format(self.weight, global_weight_min)
        assert self.weight <= global_weight_max, "edge weight {} > {}".format(self.weight, global_weight_max)

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

    def __init__(self, identifier, nodes, edges, max_num_hidden_nodes=default_max_num_hidden_nodes):

        assert len(nodes) > 0

        self.identifier = identifier
        self.nodes = deepcopy(nodes)
        self.nodes.sort(key=lambda node: node.identifier)
        self.edges = deepcopy(edges)
        self.max_num_hidden_nodes = max_num_hidden_nodes

        self.num_inputs  = len([node for node in self.nodes if node.is_input_node])
        self.num_outputs = len([node for node in self.nodes if node.is_output_node])

        self.fitness = 0
        self.possible_new_edges = []
        self.set_topology()

    @classmethod
    def default(cls, identifier, num_inputs, num_outputs, num_hidden_nodes=0, aggregation_function=default_aggregation_function, activation_function=default_activation_function, input_aggregation_function=default_input_aggregation_function, input_activation_function=default_input_activation_function, output_activation_function=default_output_activation_function, mode=default_genome_mode, weights="randomized"):

        nodes = []
        edges = []

        node_identifier = 1
        for i in range(1, num_inputs + 1):
            nodes.append(NodeGene(node_identifier, aggregation_function=input_aggregation_function, activation_function=input_activation_function, is_input_node=True))
            node_identifier += 1

        for i in range(1, num_hidden_nodes + 1):
            nodes.append(NodeGene(node_identifier, aggregation_function=aggregation_function, activation_function=activation_function))
            node_identifier += 1

        for i in range(1, num_outputs + 1):
            nodes.append(NodeGene(node_identifier, aggregation_function=aggregation_function, activation_function=output_activation_function, is_output_node=True))
            node_identifier += 1

        if mode == "fully connected":

            input_nodes  = [node for node in nodes if node.is_input_node]
            hidden_nodes = [node for node in nodes if not node.is_input_node and not node.is_output_node]
            output_nodes = [node for node in nodes if node.is_output_node]

            edge_identifier = 1
            innovation_number = 1

            if num_hidden_nodes != 0:
                for input_node in input_nodes:
                    for hidden_node in hidden_nodes:

                        if weights == "randomized":
                            weight = uniform(global_weight_min, global_weight_max)
                        else:
                            weight = 1

                        edges.append(EdgeGene(edge_identifier, innovation_number, input_node.identifier, hidden_node.identifier, weight))

                        edge_identifier   += 1
                        innovation_number += 1

                for hidden_node in hidden_nodes:
                    for output_node in output_nodes:

                        if weights == "randomized":
                            weight = uniform(global_weight_min, global_weight_max)
                        else:
                            weight = 1

                        edges.append(EdgeGene(edge_identifier, innovation_number, hidden_node.identifier, output_node.identifier, weight))
                        edge_identifier   += 1
                        innovation_number += 1
            else:
                for input_node in input_nodes:
                    for output_node in output_nodes:

                        if weights == "randomized":
                            weight = uniform(global_weight_min, global_weight_max)
                        else:
                            weight = 1

                        edges.append(EdgeGene(edge_identifier, innovation_number, input_node.identifier, output_node.identifier, weight))

                        edge_identifier   += 1
                        innovation_number += 1

        return Genome(identifier, nodes, edges)

    def set_topology(self):

        # clear topological data
        for node in self.nodes:
            node.predecessors.clear()
            node.successors.clear()
            node.layer = None

        # set immediate predecessors and successors
        for edge in self.edges:

            if edge.is_enabled:

                input_node = next(node for node in self.nodes if node.identifier == edge.input_node_identifier)
                output_node = next(node for node in self.nodes if node.identifier == edge.output_node_identifier)

                output_node.predecessors.append(input_node)
                input_node.output_nodes.append(output_node)
                input_node.successors.append(output_node)

        # set layers
        for node in [node for node in self.nodes if node.is_input_node]:
            node.layer = 1

        outer_change_occurred = True
        while outer_change_occurred:

            outer_change_occurred = False

            # forward layer determination
            change_occurred = True
            while change_occurred:
                change_occurred = False

                for node in self.nodes:

                    predecessor_layers = [predecessor.layer for predecessor in node.predecessors if predecessor.layer is not None]
                    if len(predecessor_layers) > 0:

                        new_layer = max(predecessor_layers) + 1
                        if node.layer != new_layer:
                            node.layer = new_layer
                            change_occurred = True
                            outer_change_occurred = True

            # backward layer determination
            change_occurred = True
            while change_occurred:
                change_occurred = False

                # only backwards propagate if the node behind has a bias of None (different from nonzero bias, as some activation functions do not pass through the origin)
                for node in [node for node in self.nodes if node.layer is None and node.bias is not None]:

                    successor_layers = [successor.layer for successor in node.successors if successor.layer is not None]
                    if len(successor_layers) > 0:

                        new_layer = min(successor_layers) - 1
                        if node.layer != new_layer:

                            node.layer = new_layer
                            change_occurred = True
                            outer_change_occurred = True

        # set all predecessors
        self.min_layer = min([node.layer for node in self.nodes if node.layer is not None])
        self.max_layer = max([node.layer for node in self.nodes if node.layer is not None])
        for current_layer in range(self.min_layer, self.max_layer + 1):

            # accummulate predecessors
            current_layer_nodes = [node for node in self.nodes if node.layer == current_layer]
            for current_layer_node in current_layer_nodes:
                for predecessor in current_layer_node.predecessors:

                    current_layer_node.predecessors += [pre_predecessor for pre_predecessor in predecessor.predecessors if pre_predecessor not in current_layer_node.predecessors]

        # set all successors
        for current_layer in range(self.max_layer, self.min_layer - 1, -1):

            #accummulate successors
            current_layer_nodes = [node for node in self.nodes if node.layer == current_layer]
            for current_layer_node in current_layer_nodes:
                for successor in current_layer_node.successors:

                    current_layer_node.successors += [post_successor for post_successor in successor.successors]# if post_successor not in current_layer_node.successors]

        for node in self.nodes:
            node.predecessors.sort(key=lambda predecessor : predecessor.identifier)
            node.successors.sort(key=lambda successor : successor.identifier)

    def mutate_add_node(self, new_node_identifier=None, innovation_number_1=None, innovation_number_2=None, disabled_edge_identifier=None, mode=None, new_node_aggregation_function=default_aggregation_function, new_node_activation_function=default_activation_function):

        if new_node_identifier == None:
            new_node_identifier = max([node.identifier for node in self.nodes] + [0]) + 1

        active_edges = [edge for edge in self.edges if edge.is_enabled]

        if len(active_edges) > 0:

            if disabled_edge_identifier == None:
                # choose a random active edge
                disabled_edge = choice(active_edges)
                disabled_edge.is_enabled = False
            else:
                disabled_edge = [edge for edge in active_edges if edge.identifier == disabled_edge_identifier][0]

            if mode == "random":
                new_node_aggregation_function = choice(aggregation_function_names)
                new_node_activation_function = choice(activation_function_names)

            # create a new node
            new_node = NodeGene(new_node_identifier, aggregation_function=new_node_aggregation_function, activation_function=new_node_activation_function)

            # create two new edges
            new_edge_identifier = max([edge.identifier for edge in self.edges] + [0]) + 1
            new_edge_1 = EdgeGene(new_edge_identifier, innovation_number_1, disabled_edge.input_node_identifier, new_node.identifier, 1)
            new_edge_2 = EdgeGene(new_edge_identifier + 1, innovation_number_2, new_node.identifier, disabled_edge.output_node_identifier, disabled_edge.weight)

            self.nodes.append(new_node)
            self.edges.append(new_edge_1)
            self.edges.append(new_edge_2)

            self.set_topology()

            return new_node

        return False

    def mutate_remove_node(self, removed_node_identifier=None):

        if removed_node_identifier == None:
            # choose a random node to remove
            removed_node = choice([node for node in self.nodes if not node.is_input_node and not node.is_output_node and node.is_enabled] + [None])
        else:
            removed_node = ([node for node in self.nodes if not node.is_input_node and not node.is_output_node and node.is_enabled and node.identifier == removed_node_identifier] + [None])[0]

        if removed_node is not None:
            # disable the relevant node
            removed_node.is_enabled = False

            # get all the enabled edges that use the removed node as input or output and disable them
            removed_edges = [edge for edge in self.edges if edge.is_enabled and (edge.input_node_identifier == removed_node.identifier or edge.output_node_identifier == removed_node.identifier)]
            for edge in removed_edges:
                edge.is_enabled = False

        self.set_topology()

    def mutate_set_bias(self, node=None, bias=None):

        if node == None:
            node = choice([node for node in self.nodes if not node.is_input_node and not node.is_output_node])

        if bias == None:
            new_bias = uniform(global_bias_min, global_bias_max)
        else:
            new_bias = bias

        node.bias = new_bias

    def mutate_add_edge(self, new_edge_identifier=None, input_node_identifier=None, output_node_identifier=None, weight=None, weight_min=global_weight_min, weight_max=global_weight_max):

        self.set_topology()

        if new_edge_identifier == None:
            new_edge_identifier = max([edge.identifier for edge in self.edges] + [0]) + 1

        possible_input_nodes  = [node for node in self.nodes if not node.is_output_node and node.is_enabled]
        possible_output_nodes = [node for node in self.nodes if not node.is_input_node and node.is_enabled]

        self.possible_new_edges.clear()
        for possible_input_node in possible_input_nodes:
            for possible_output_node in possible_output_nodes:

                if (possible_input_node is not possible_output_node) and (possible_output_node not in possible_input_node.predecessors) and (possible_output_node not in possible_input_node.output_nodes):

                    # print("possible input node {} successors : {}, new node: {}".format(possible_input_node.identifier, [node.identifier for node in possible_input_node.successors], possible_output_node.identifier))
                    # if (possible_output_node in possible_input_node.successors):
                    #     raise RuntimeError

                    if weight == None:
                        weight = uniform(weight_min, weight_max)
                    self.possible_new_edges.append(EdgeGene(new_edge_identifier, None, possible_input_node.identifier, possible_output_node.identifier, weight))

        if len(self.possible_new_edges) > 0:

            new_edge = deepcopy(choice(self.possible_new_edges))

            self.edges.append(new_edge)
            self.set_topology()

        else:
            new_edge = None
        #     print("attempted to add edge when no possible edges were available")
        #
        # print("added edge {}\n".format(new_edge))

        return new_edge

    def mutate_remove_edge(self, removed_edge_identifier=None):

        if removed_edge_identifier is not None:
            removed_edge = ([edge for edge in self.edges if edge.identifier == removed_edge_identifier] + [None])[0]
        else:
            removed_edge = choice(self.edges + [None])

        if removed_edge is not None:
            removed_edge.is_enabled = False

        self.set_topology()

        return removed_edge

    def mutate_reset_weight(self, reset_edge_identifier=None, weight_minimum=initial_weight_min, weight_maximum=initial_weight_max):

        if reset_edge_identifier == None:
            reset_edge = choice([edge for edge in self.edges] + [None])
        else:
            reset_edge = ([edge for edge in self.edges if edge.identifier == reset_edge_identifier] + [None])[0]

        if reset_edge is not None:
            new_weight = uniform(weight_minimum, weight_maximum)
            reset_edge.weight = new_weight

    def mutate_scale_weight(self, mutated_edge_identifier=None, scale_min=weight_scale_min, scale_max=weight_scale_max, weight_minimum=global_weight_min, weight_maximum=global_weight_max):

        if mutated_edge_identifier == None:
            mutated_edge = choice([edge for edge in self.edges] + [None])
        else:
            mutated_edge = ([edge for edge in self.edges if edge.identifier == mutated_edge_identifier] + [None])[0]

        if mutated_edge is not None:
            new_weight = mutated_edge.weight * uniform(scale_min, scale_max)
            new_weight = max(weight_minimum, new_weight)
            new_weight = min(weight_maximum, new_weight)

            mutated_edge.weight = new_weight

    def mutate_change_aggregation_function(self, mutated_node_identifier=None):

        if mutated_node_identifier == None:
            mutated_node = choice([node for node in self.nodes if not node.is_input_node and not node.is_output_node and node.is_enabled] + [None])
        else:
            mutated_node = choice(([node for node in self.nodes if not node.is_input_node and not node.is_output_node and node.is_enabled and node.identifier == mutated_node_identifier] + [None])[0])

        if mutated_node is not None:
            new_aggregation_function = choice(aggregation_function_names)
            mutated_node.aggregation_function = new_aggregation_function

    def mutate_change_activation_function(self, mutated_node_identifier=None):

        if mutated_node_identifier == None:
            mutated_node = choice([node for node in self.nodes if not node.is_input_node and not node.is_output_node and node.is_enabled] + [None])
        else:
            mutated_node = choice(([node for node in self.nodes if not node.is_input_node and not node.is_output_node and node.is_enabled and node.identifier == mutated_node_identifier] + [None])[0])

        if mutated_node is not None:
            new_activation_function = choice(activation_function_names)
            mutated_node.activation_function = new_activation_function

    def random_mutation(self):

        # choose a mutation
        random_number = uniform(0, 1)
        possible_mutations = [mutation for mutation, probability in mutations.items() if random_number < probability]
        mutation = choice(possible_mutations + [None])

        if mutation == "add node" and self.num_hidden_nodes() < self.max_num_hidden_nodes:
            mutation = self.mutate_add_node()
            # have to set innovation numbers after this
            # this is done on the population level

        elif mutation == "remove node" and self.num_hidden_nodes() > 0:
            self.mutate_remove_node()

        elif mutation == "set bias" and self.num_hidden_nodes() > 0:
            self.mutate_set_bias()

        elif mutation == "add edge":
            mutation = self.mutate_add_edge()
            # have to set innovation numbers after this
            # this is done on the population level

        elif mutation == "remove edge" and self.num_edges() > 0:
            self.mutate_remove_edge()

        elif mutation == "reset weight" and self.num_edges() > 0:
            self.mutate_reset_weight()

        elif mutation == "scale weight" and self.num_edges() > 0:
            self.mutate_scale_weight()

        elif mutation == "change aggregation function" and self.num_hidden_nodes() > 0:
            self.mutate_change_aggregation_function()

        elif mutation == "change activation function" and self.num_hidden_nodes() > 0:
            self.mutate_change_activation_function()

        else:
            mutation = None
            # raise ValueError("Incorrect mutation requested: " + mutation)

        return mutation

    @classmethod
    def crossover(cls, genome1, genome2, new_genome_identifier):

        better_genome = None
        worse_genome = None

        # if self Genome performs better than other_genome
        if genome1.fitness > genome2.fitness:
            better_genome = genome1
            worse_genome = genome2
        else:
            better_genome = genome2
            worse_genome = genome1

        # crossover edges
        better_genome_edges = sorted(better_genome.edges, key=lambda edge : edge.innovation_number)
        worse_genome_edges = sorted(worse_genome.edges, key=lambda edge : edge.innovation_number)
        worse_genome_edge_innovation_numbers = [edge.innovation_number for edge in worse_genome_edges]


        new_genome_edges = []
        for better_genome_edge in better_genome_edges:
            if better_genome_edge.innovation_number in worse_genome_edge_innovation_numbers:
                worse_genome_edge = [edge for edge in worse_genome_edges if better_genome_edge.innovation_number == edge.innovation_number][0]

                new_edge = deepcopy(choice([better_genome_edge, worse_genome_edge]))
                new_genome_edges.append(new_edge)

        new_genome_node_identifiers = []
        new_genome_node_identifiers += [edge.input_node_identifier for edge in new_genome_edges if edge.input_node_identifier not in new_genome_node_identifiers]
        new_genome_node_identifiers += [edge.output_node_identifier for edge in new_genome_edges if edge.output_node_identifier not in new_genome_node_identifiers]

        hidden_nodes = deepcopy(better_genome.nodes) + deepcopy(worse_genome.nodes)
        better_genome_node_identifiers = [node.identifier for node in better_genome.nodes]
        worse_genome_node_identifiers = [node.identifier for node in worse_genome.nodes]

        new_genome_nodes = [node for node in better_genome.nodes if node.is_input_node or node.is_output_node]
        for new_node_identifier in new_genome_node_identifiers:

            # next((x for x in test_list if x.value == value), None)
            instance_from_better_genome = next((node for node in better_genome.nodes if node.identifier == new_node_identifier), None)
            instance_from_worse_genome = next((node for node in worse_genome.nodes if node.identifier == new_node_identifier), None)

            if instance_from_worse_genome is None:
                new_node = instance_from_better_genome
            elif instance_from_better_genome is None:
                new_node = instance_from_worse_genome
            elif instance_from_better_genome.is_enabled == False:
                new_node = instance_from_better_genome
            elif instance_from_worse_genome.is_enabled == False:
                new_node = instance_from_worse_genome
            else:
                new_node = instance_from_better_genome

            if new_node not in new_genome_nodes:
                new_genome_nodes.append(new_node)

        new_genome = Genome(new_genome_identifier, new_genome_nodes, new_genome_edges)

        assert len([node for node in new_genome.nodes if node.is_input_node]) == len([node for node in genome1.nodes if node.is_input_node])
        assert len([node for node in new_genome.nodes if node.is_output_node]) == len([node for node in genome1.nodes if node.is_output_node])

        return new_genome

    def num_hidden_nodes(self):

        return len([node for node in self.nodes if not node.is_input_node and not node.is_output_node and node.is_enabled])

    def num_edges(self):

        return len(self.edges)

    def save(self, filename):

        file = open(filename, "wb")
        pickle.dump(self, file, protocol=-1)

    @classmethod
    def from_file(cls, filename):
        file = open(filename, "rb")
        return pickle.load(file)

    def copy(self):
        return deepcopy(self)

    def similarity(self, genome):

        numerator   = 0
        denominator = 0

        node_identifiers = [node.identifier for node in self.nodes]
        other_node_identifiers = [node.identifier for node in genome.nodes]

        edge_innovation_numbers = [edge.innovation_number for edge in self.edges]
        other_edge_innovation_numbers = [edge.innovation_number for edge in genome.edges]

        for node_identifier in node_identifiers:
            if node_identifier in other_node_identifiers:
                numerator += node_gene_similarity_coefficient

        denominator += node_gene_similarity_coefficient * max(len(self.nodes), len(genome.nodes))

        for edge_innovation_number in edge_innovation_numbers:
            if edge_innovation_number in other_edge_innovation_numbers:
                numerator += edge_gene_similarity_coefficient

        denominator += edge_gene_similarity_coefficient * max(len(self.edges), len(genome.edges))

        return numerator / denominator

    def __str__(self):

        representation = "Genome {}, fitness {}:\n".format(self.identifier, self.fitness)

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