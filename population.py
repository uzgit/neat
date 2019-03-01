from random import *
import sys

from globals import *
from species import *
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

    def __init__(self, population_size, num_generations, num_inputs, num_outputs, initial_num_hidden_nodes=0, max_num_hidden_nodes=default_max_num_hidden_nodes, output_activation_function=default_output_activation_function, mode="unconnected", output_stream=sys.stdout):

        self.next_innovation_number = 1
        ####################################################################
        self.innovations = []
        ####################################################################
        self.next_genome_identifier = 1
        self.next_species_identifier = 1

        self.population_size = population_size
        self.num_generations = num_generations
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.initial_num_hidden_nodes = initial_num_hidden_nodes
        self.max_num_hidden_nodes = max_num_hidden_nodes
        self.mode = mode
        self.output_activation_function = output_activation_function

        self.output_stream = output_stream

        self.genomes = []
        self.species = []
        self.misfits = []
        self.neural_networks = []

        self.champion = None

    def initialize_genomes(self):

        for i in range(self.population_size):

            self.misfits.append(Genome.default(self.next_genome_identifier, self.num_inputs, self.num_outputs, self.initial_num_hidden_nodes, output_activation_function=self.output_activation_function, mode=self.mode))
            self.next_genome_identifier += 1

    def initial_mutation(self):

        for genome in self.misfits:
            self.mutate_genome(genome)

    def mutate_genome(self, genome):

        result = genome.random_mutation()

        if isinstance(result, NodeGene):

            new_edges = [edge for edge in genome.edges if edge.innovation_number == None]
            for edge in new_edges:
                self.set_innovation_number(edge)

        elif isinstance(result, EdgeGene):

            self.set_innovation_number(result)

    def run_with_local_fitness_function(self, evaluation_function, num_generations=None, fitness_goal=None, output_stream=None):

        if num_generations == None:
            num_generations = self.num_generations

        if output_stream is None:
            output_stream = self.output_stream

        print("Beginning run: {} members, {} generations, {} fitness goal.".format(self.population_size, num_generations, fitness_goal), file=output_stream)

        self.initialize_genomes() # all genomes are either unconnected or fully connected
        for i in range(10):
            self.initial_mutation()
        self.set_species()
        # self.mutate_all_genomes() # randomize the genomes a bit

        self.champion = deepcopy(self.genomes[0])

        self.generations_run = 0
        while (self.generations_run < num_generations if num_generations != -1 else True) and (self.max_fitness() < fitness_goal if fitness_goal is not None else True):

            print("Beginning generation {} with {} individuals of {} species.".format(self.generations_run + 1, self.size(), self.num_species()))

            # self.set_species()
            self.remove_empty_species()

            # self.neural_networks.clear()
            self.set_neural_networks()
            self.evaluate_neural_networks(evaluation_function)
            self.set_species_fitnesses()

            # report species
            #######################################################################################
            for i in range(6 + 6 + 10 + 15):
                print("-", end="", file=output_stream)
            print(file=output_stream)
            print("%-6s%6s%10s%14s" % ("Species", "Age", "Members", "Fitness"), file=output_stream)
            for i in range(6 + 6 + 10 + 15):
                print("-", end="", file=output_stream)
            print(file=output_stream)
            for species in self.species:
                print(species.information_entry(), file=output_stream)
            #######################################################################################

            # post-evalution metadata and cleanup
            generation_champion = self.get_generation_champion()

            self.step_generation()

            # update loop conditions
            self.generations_run += 1

            if generation_champion.fitness >= self.champion.fitness:
                self.champion = generation_champion
                print("Best genome in generation {}: genome {}, fitness: {}".format(self.generations_run, generation_champion.identifier, generation_champion.fitness),file=output_stream, end="\n\n")
            # else:
            #     raise RuntimeError("Serious error here.")

            for genome in self.genomes:
                for edge in genome.edges:
                    if edge.innovation_number == None:
                        self.set_innovation_number(edge)

        return self.champion

    def step_generation(self):

        self.genomes.clear()
        if self.num_species() > 1:
            self.remove_stagnated_species()
        self.set_total_fitness()
        self.step_species_generation()
        self.species_reproduce()
        self.set_species()

    # assume empty species but full genome list
    def set_species(self):

        # assign all uncategorized genomes to species
        for genome in self.misfits:

            compatible_species = ([species for species in self.species if species.is_compatible_with(genome)] + [None])[0]
            if compatible_species is None:
                new_species = Species(self.next_species_identifier, 1, genome)
                self.species.append( new_species )
                self.next_species_identifier += 1

                new_species.add_genome(genome)

                # print("adding genome {} to newly created species {}".format(genome.identifier, new_species.identifier))

            else:
                compatible_species.add_genome(genome)
                # print("adding genome {} to species {}".format(genome.identifier, compatible_species.identifier))

            self.genomes.append(genome)

        self.misfits.clear()

        # ensure all species contain at least 2 genomes
        for species in self.species:
            while species.size() < 2:

                new_genome = deepcopy(species.champion)
                new_genome.identifier = self.next_genome_identifier
                self.next_genome_identifier += 1

                # Genome.random_mutation does not set innovation_number
                # new_genome.random_mutation()
                self.mutate_genome(new_genome)

                species.add_genome(new_genome)
                self.genomes.append(new_genome)

                # print("necessarily added genome {} to species {}".format(new_genome.identifier, species.identifier))

    def remove_empty_species(self):

        for species in self.species:
            if species.size() == 0:
                self.species.remove(species)

    def set_species_fitnesses(self):

        for species in self.species:
            species.add_fitness( max([genome.fitness for genome in species.genomes] ) )

    def remove_stagnated_species(self):

        for species in self.species:

            if species.is_stagnated() or species.size() == 0:

                print("Removing stagnated species {}.".format(species.identifier), file=self.output_stream)
                self.species.remove(species)

    def step_species_generation(self):

        self.set_total_fitness()

        for species in self.species:

            species.step_generation()

            num_individuals = round(self.population_size * species.fitness / self.total_fitness)
            # print("args: self.population_size={}, species.fitness={}, self.total_fitness={}, raw_num_children={}, num_children={}".format(self.population_size, species.fitness, self.total_fitness, round(self.population_size * species.fitness / self.total_fitness), num_individuals))
            children_generated = species.reproduce(num_individuals, self.next_genome_identifier)
            self.next_genome_identifier = children_generated

            for genome in species.elites:
                self.genomes.append(genome)

            # print("len species.children", len(species.genomes))
            for genome in species.genomes:
                # self.mutate_genome(genome)
                self.genomes.append(genome)
                # print("added genome {}".format(genome.identifier))

    def species_reproduce(self):

        total_children_requested = 0

        # num_children_array = []
        # for species in self.species:
        #     num_children_array.append(round(self.population_size * species.fitness / self.total_fitness))
        # print("num children array:", num_children_array)

        for species in self.species:

            num_children = round(self.population_size * species.fitness / self.total_fitness)
            total_children_requested += num_children
            # print("args: self.population_size={}, species.fitness={}, self.total_fitness={}, raw_num_children={}, num_children={}".format(self.population_size, species.fitness, self.total_fitness, round(self.population_size * species.fitness / self.total_fitness), num_children))
            children_generated = species.reproduce(num_children, self.next_genome_identifier)
            self.next_genome_identifier = children_generated

            for genome in species.elites:
                self.genomes.append(genome)

            for genome in species.genomes:
                self.genomes.append(genome)

            for genome in species.misfits:
                self.misfits.append(genome)

        # print("total_children_requested", total_children_requested)

    def set_neural_networks(self):

        self.neural_networks.clear()
        for genome in self.genomes:
            genome.fitness = 0
            neural_network = FeedForwardNeuralNetwork(genome)
            self.neural_networks.append(neural_network)

    def evaluate_neural_networks(self, evaluation_function):

        for neural_network in self.neural_networks:
            neural_network.genome.fitness = evaluation_function(neural_network)

    def set_innovation_number(self, edge):

        matching_edge = ([matching_edge for matching_edge in self.innovations if edge.input_node_identifier == matching_edge.input_node_identifier and edge.output_node_identifier == matching_edge.output_node_identifier] + [None])[0]
        if matching_edge == None:

            edge.innovation_number = self.next_innovation_number

            self.innovations.append(edge)
            self.next_innovation_number += 1

        else:
            edge.innovation_number = matching_edge.innovation_number

    def max_fitness(self):

        return max([genome.fitness for genome in self.genomes])

    def set_total_fitness(self):

        total_fitness = sum( [species.average_fitness() for species in self.species] )

        if total_fitness == 0:
            total_fitness = 1

        self.total_fitness = total_fitness

    def get_generation_champion(self):

        self.genomes.sort(key=lambda genome: genome.fitness, reverse=True)
        champion = deepcopy([genome for genome in self.genomes][0])

        # print(champion)

        return champion

    def size(self):

        return len(self.genomes)

    def num_species(self):

        return len(self.species)

    def __str__(self):

        return "Population of {} individuals. {} inputs, {} outputs".format(len(self.genomes), self.num_inputs, self.num_outputs)

    # def run_with_local_fitness_function_deprecated(self, evaluation_function, num_generations=None, fitness_goal=None, output_stream=None):
    #
    #     if num_generations == None:
    #         num_generations = self.num_generations
    #
    #
    #
    #     self.initialize_genomes()
    #
    #     self.generations_run = 0
    #     while (self.generations_run < num_generations if num_generations != -1 else True) and (max_fitness < fitness_goal if fitness_goal is not None else True):
    #
    #         # mutate genomes if we have passed the first run
    #         if self.generations_run > 1:
    #             self.mutate_all_genomes()
    #
    #         # create neural networks
    #         self.neural_networks.clear()
    #         for genome in self.genomes:
    #             genome.fitness = 0
    #
    #             neural_network = FeedForwardNeuralNetwork(genome)
    #             self.neural_networks.append(neural_network)
    #
    #         for neural_network in self.neural_networks:
    #             evaluation_function(neural_network)
    #
    #         # update loop conditions
    #         max_fitness = max([genome.fitness for genome in self.genomes])
    #         self.generations_run += 1
    #
    #         # find best genome
    #         sorted_genomes = sorted(self.genomes, key=lambda genome : genome.fitness, reverse=True)
    #         print("Best genome in generation {}: genome {}, fitness: {}".format(self.generations_run, sorted_genomes[0].identifier, sorted_genomes[0].fitness), file=output_stream)
    #
    #     return sorted_genomes[0]

    # def mutate_genome_deprecated(self, genome):
    #
    #     num_mutations = 0
    #
    #     while num_mutations < max_num_mutations_per_individual_per_generation:
    #
    #         successfully_mutated = True
    #
    #         # choose a mutation
    #         random_number = uniform(0, 1)
    #         possible_mutations = [mutation for mutation, probability in mutations.items() if random_number < probability]
    #         mutation = choice(possible_mutations + [None])
    #
    #         if mutation == None or mutation == "":
    #             successfully_mutated = False
    #
    #         elif mutation == "add node" and genome.num_hidden_nodes() < self.max_num_hidden_nodes:
    #             node_identifier = max([node.identifier for node in genome.nodes]) + 1
    #             new_edge_1, new_edge_2 = genome.mutate_add_node(node_identifier, self.next_innovation_number, self.next_innovation_number + 1)
    #
    #             if new_edge_1 is not None:
    #                 self.set_innovation_number(new_edge_1)
    #                 self.set_innovation_number(new_edge_2)
    #
    #         elif mutation == "remove node":
    #             genome.mutate_remove_node()
    #
    #         elif mutation == "add edge":
    #             edge_identifier = max([edge.identifier for edge in genome.edges], default=0) + 1
    #             edge = genome.mutate_add_edge(edge_identifier)
    #             if edge is not None:
    #                 self.set_innovation_number(edge)
    #             else: #if the genome is already fully connected
    #                 successfully_mutated = False
    #
    #         elif mutation == "remove edge":
    #             genome.mutate_remove_edge()
    #
    #         elif mutation == "reset weight":
    #             genome.mutate_reset_weight()
    #
    #         elif mutation == "scale weight":
    #             genome.mutate_scale_weight()
    #
    #         elif mutation == "change aggregation function":
    #             genome.mutate_change_aggregation_function()
    #
    #         elif mutation == "change activation function":
    #             genome.mutate_change_activation_function()
    #
    #         else:
    #             successfully_mutated = False
    #             raise ValueError("Incorrect mutation requested: " + mutation)
    #
    #         #if successfully_mutated:
    #         num_mutations += 1
    #
