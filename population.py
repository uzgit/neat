from random import *

from globals import *
from genome import *
from neural_network import *

# I would arrange the dictionary with keys and values reversed, but dictionary keys must be unique
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

class Population:

    def __init__(self, population_size, num_inputs, num_outputs, initial_num_hidden_nodes=0, mode="unconnected"):

        self.innovation_number = 1
        ####################################################################
        self.innovations = []
        ####################################################################
        self.genome_identifier = 1

        self.population_size = population_size
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.initial_num_hidden_nodes = initial_num_hidden_nodes

        self.genomes = []
        self.species = []

        self.initialize_genomes()

    def initialize_genomes(self):

        for i in range(self.population_size):

            self.genomes.append(Genome.default(self.genome_identifier, self.num_inputs, self.num_outputs, self.initial_num_hidden_nodes))

    def mutate_all_genomes(self):

        for genome in self.genomes:
            self.mutate_genome(genome)

    def mutate_genome(self, genome):

        num_mutations = 0

        while num_mutations < max_num_mutations_per_individual_per_generation:

            successfully_mutated = True

            # choose a mutation
            random_number = uniform(0, 1)
            possible_mutations = [mutation for mutation, probability in mutations.items() if random_number < probability]
            mutation = choice(possible_mutations + [None])

            if mutation == None or mutation == "":
                successfully_mutated = False

            elif mutation == "add node":
                node_identifier = max([node.identifier for node in genome.nodes]) + 1
                genome.mutate_add_node(node_identifier, self.innovation_number + 1, self.innovation_number + 2)
                self.innovation_number += 2

            elif mutation == "remove node":
                genome.mutate_remove_node()

            elif mutation == "add edge":
                edge_identifier = max([edge.identifier for edge in genome.edges], default=0) + 1
                edge = genome.mutate_add_edge(edge_identifier)
                if edge is not None:
                    fsdfsdfds = 1
                    # deal with innovation number here
                else: #if the genome is already fully connected
                    successfully_mutated = False

            elif mutation == "remove edge":
                genome.mutate_remove_edge()

            elif mutation == "reset weight":
                genome.mutate_reset_weight()

            elif mutation == "scale weight":
                genome.mutate_scale_weight()

            elif mutation == "change aggregation function":
                genome.mutate_change_aggregation_function()

            elif mutation == "change activation function":
                genome.mutate_change_activation_function()

            else:
                successfully_mutated = False
                raise ValueError("Incorrect mutation requested: " + mutation)

            if successfully_mutated:
                num_mutations += 1

    def __str__(self):

        return "Population of {} individuals. {} inputs, {} outputs".format(len(self.genomes), self.num_inputs, self.num_outputs)