from copy import deepcopy
from random import *
import pickle

from functions import *
from globals import *

mutations = {
    "add node" : mutate_add_node_probability,
    "remove node" : mutate_remove_node_probability,
    "add edge" : mutate_add_edge_probability,
    "remove edge" : mutate_remove_edge_probability,
    "reset weight" : mutate_reset_weight_probability,
    "scale weight" : mutate_scale_weight_probability,
    "change aggregation function" : mutate_change_aggregation_function_probability,
    "change activation function" : mutate_change_activation_function_probability,
}

class NodeGene:

    def __init__(self, identifier, aggregation_function=default_aggregation_function, activation_function=default_activation_function, is_input_node=False, is_output_node=False, is_enabled=True):

        self.identifier = identifier
        self.aggregation_function = aggregation_function
        self.activation_function = activation_function
        self.is_input_node= is_input_node
        self.is_output_node = is_output_node

        self.is_enabled = is_enabled

    def __str__(self):

        representation = "node {}: {}, {}".format(self.identifier, self.aggregation_function, self.activation_function)

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

    def mutate_add_node(self, new_node_identifier=None, innovation_number_1=None, innovation_number_2=None, disabled_edge_identifier=None, mode=None, new_node_aggregation_function=default_aggregation_function, new_node_activation_function=default_activation_function):

        if new_node_identifier == None:
            new_node_identifier = max([node.identifier for node in self.nodes] + [0]) + 1

        active_edges = [edge for edge in self.edges if edge.is_enabled]

        # new_node = None

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
            new_node = NodeGene(new_node_identifier, new_node_aggregation_function, new_node_activation_function)

            # create two new edges
            new_edge_identifier = max([edge.identifier for edge in self.edges] + [0]) + 1
            new_edge_1 = EdgeGene(new_edge_identifier, innovation_number_1, disabled_edge.input_node_identifier, new_node.identifier, 1)
            new_edge_2 = EdgeGene(new_edge_identifier + 1, innovation_number_2, new_node.identifier, disabled_edge.output_node_identifier, disabled_edge.weight)

            self.nodes.append(new_node)
            self.edges.append(new_edge_1)
            self.edges.append(new_edge_2)

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

    def mutate_add_edge(self, new_edge_identifier=None, input_node_identifier=None, output_node_identifier=None, weight=None, weight_min=global_weight_min, weight_max=global_weight_max):

        if new_edge_identifier == None:
            new_edge_identifier = max([edge.identifier for edge in self.edges] + [0]) + 1

        new_edge = False

        # for scope
        input_node  = None
        output_node = None

        if input_node_identifier == None:

            possible_input_nodes = [node for node in self.nodes if not node.is_output_node and node.is_enabled]
            shuffle(possible_input_nodes)

            found_edge = False
            i = 0
            while i < len(possible_input_nodes) and found_edge == False:

                predecessors = []

                possible_output_node = None

                # get the edges leading into the possible input node
                possible_input_node = possible_input_nodes[i]
                input_edge_stack = [edge for edge in self.edges if edge.output_node_identifier == possible_input_node.identifier]
                while(len(input_edge_stack) > 0):
                    new_input_edge_stack = []
                    for input_edge in input_edge_stack:

                        local_input_nodes = [node for node in self.nodes if node.identifier == input_edge.input_node_identifier and node.is_enabled]
                        if len(local_input_nodes) > 0:
                            predecessor = local_input_nodes[0]
                            predecessors.append(predecessor)

                            new_input_edge_stack += [edge for edge in self.edges if edge.output_node_identifier == predecessor.identifier]

                    input_edge_stack = [] + new_input_edge_stack

                possible_output_nodes = [node for node in self.nodes if node not in predecessors and not node.is_input_node and node.is_enabled and node.identifier != possible_input_node.identifier]
                shuffle(possible_output_nodes)

                found_output_node = False
                ii = 0
                while ii < len(possible_output_nodes) and found_output_node == False:

                    possible_output_node = possible_output_nodes[ii]

                    # check for duplicates
                    if len([edge for edge in self.edges if edge.input_node_identifier != possible_input_node.identifier and edge.output_node_identifier != possible_output_node.identifier]) == 0:
                        output_node = possible_output_node
                        found_output_node = True

                    ii += 1

                if possible_input_node is not None and possible_output_node is not None:

                    input_node = possible_input_node
                    output_node = possible_output_node

                    found_edge = True

                i += 1

        else:
            input_node  = [node for node in self.nodes if node.is_enabled and node.identifier == input_node_identifier][0]
            output_node = [node for node in self.nodes if node.is_enabled and node.identifier == output_node_identifier][0]

        if weight == None:
            weight = uniform(weight_min, weight_max)

        # check if the edge already exists and is enabled
        edge_exists_already = any([edge.input_node_identifier == input_node.identifier and edge.output_node_identifier == output_node.identifier for edge in self.edges])
        if edge_exists_already:
            new_edge = None
        else:
            new_edge = EdgeGene(new_edge_identifier, None, input_node.identifier, output_node.identifier, weight)

        if new_edge is not None:
            self.edges.append(new_edge)

        return new_edge

    def mutate_remove_edge(self, removed_edge_identifier=None):

        if removed_edge_identifier is not None:
            removed_edge = ([edge for edge in self.edges if edge.identifier == removed_edge_identifier] + [None])[0]
        else:
            removed_edge = choice(self.edges + [None])

        if removed_edge is not None:
            removed_edge.is_enabled = False

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

        elif mutation == "remove node" and self.num_hidden_nodes() > 0:
            self.mutate_remove_node()

        elif mutation == "add edge":
            mutation = self.mutate_add_edge()

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
        # print("in crossover: better_genome.edges = {}".format(better_genome.edges))
        better_genome_edges = sorted(better_genome.edges, key=lambda edge : edge.innovation_number)
        # better_genome_edge_innovation_numbers = [edge.innovation_number for edge in better_genome_edges]
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

        # we must include all input and output nodes
        # new_genome_nodes = []
        # for node in [node for node in better_genome.nodes if node.is_input_node or node.is_output_node]:
        #     new_genome_nodes.append(node)
        #
        # for new_node_identifier in new_genome_node_identifiers:
        #
        #     new_node = None
        #
        #     # if both genomes contain the node
        #     if new_node_identifier in better_genome_node_identifiers and new_node_identifier in worse_genome_node_identifiers:
        #
        #         matching_node_1 =
        #
        #         if uniform(0, 1) < 0.5:
        #             new_node = deepcopy([node for node in better_genome.nodes if node.identifier == new_node_identifier][0])
        #         else:
        #             new_node = deepcopy([node for node in worse_genome.nodes if node.identifier == new_node_identifier][0])
        #
        #     else:
        #         new_node = deepcopy([node for node in all_nodes if node.identifier == new_node_identifier][0])
        #
        #     # if new_node not in new_genome_nodes:
        #     new_genome_nodes.append(new_node)
        #
        # new_genome = Genome(new_genome_identifier, new_genome_nodes, new_genome_edges)

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