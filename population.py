from random import *

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

    def __init__(self, population_size, num_generations, num_inputs, num_outputs, initial_num_hidden_nodes=0, max_num_hidden_nodes=10, output_activation_function=default_output_activation_function, mode="unconnected"):

        self.next_innovation_number = 1
        ####################################################################
        self.innovations = []
        ####################################################################
        self.genome_identifier = 1
        self.next_species_identifier = 1

        self.population_size = population_size
        self.num_generations = num_generations
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.initial_num_hidden_nodes = initial_num_hidden_nodes
        self.max_num_hidden_nodes = max_num_hidden_nodes
        self.mode = mode
        self.output_activation_function = output_activation_function

        self.genomes = []
        self.species = []
        self.neural_networks = []

        #self.champion = None

    def initialize_genomes(self):

        for i in range(self.population_size):

            self.genomes.append(Genome.default(self.genome_identifier, self.num_inputs, self.num_outputs, self.initial_num_hidden_nodes, output_activation_function=self.output_activation_function, mode=self.mode))
            self.genome_identifier += 1

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

            elif mutation == "add node" and genome.num_hidden_nodes() < self.max_num_hidden_nodes:
                node_identifier = max([node.identifier for node in genome.nodes]) + 1
                new_edge_1, new_edge_2 = genome.mutate_add_node(node_identifier, self.next_innovation_number, self.next_innovation_number + 1)

                if new_edge_1 is not None:
                    self.set_innovation_number(new_edge_1)
                    self.set_innovation_number(new_edge_2)

            elif mutation == "remove node":
                genome.mutate_remove_node()

            elif mutation == "add edge":
                edge_identifier = max([edge.identifier for edge in genome.edges], default=0) + 1
                edge = genome.mutate_add_edge(edge_identifier)
                if edge is not None:
                    self.set_innovation_number(edge)
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

            #if successfully_mutated:
            num_mutations += 1

    def run_with_local_fitness_function_deprecated(self, evaluation_function, num_generations=None, fitness_goal=None, output_stream=None):

        if num_generations == None:
            num_generations = self.num_generations

        print("Beginning run: {} members, {} generations, {} fitness goal.".format(self.population_size, num_generations, fitness_goal), file=output_stream)

        self.initialize_genomes()

        max_fitness = max([genome.fitness for genome in self.genomes])
        self.generations_run = 0
        while (self.generations_run < num_generations if num_generations != -1 else True) and (max_fitness < fitness_goal if fitness_goal is not None else True):

            # mutate genomes if we have passed the first run
            if self.generations_run > 1:
                self.mutate_all_genomes()

            # create neural networks
            self.neural_networks.clear()
            for genome in self.genomes:
                genome.fitness = 0

                neural_network = FeedForwardNeuralNetwork(genome)
                self.neural_networks.append(neural_network)

            for neural_network in self.neural_networks:
                evaluation_function(neural_network)

            # update loop conditions
            max_fitness = max([genome.fitness for genome in self.genomes])
            self.generations_run += 1

            # find best genome
            sorted_genomes = sorted(self.genomes, key=lambda genome : genome.fitness, reverse=True)
            print("Best genome in generation {}: genome {}, fitness: {}".format(self.generations_run, sorted_genomes[0].identifier, sorted_genomes[0].fitness), file=output_stream)

        return sorted_genomes[0]

    def run_with_local_fitness_function(self, evaluation_function, num_generations=None, fitness_goal=None, output_stream=None):

        if num_generations == None:
            num_generations = self.num_generations

        print("Beginning run: {} members, {} generations, {} fitness goal.".format(self.population_size, num_generations, fitness_goal), file=output_stream)

        self.initialize_genomes() # all genomes are either unconnected or fully connected
        self.mutate_all_genomes() # randomize the genomes a bit

        self.champion = deepcopy(self.genomes[0])

        max_fitness = 0
        self.generations_run = 0
        while (self.generations_run < num_generations if num_generations != -1 else True) and (max_fitness < fitness_goal if fitness_goal is not None else True):

            self.set_species()

            self.neural_networks.clear()
            for genome in self.genomes:
                genome.fitness = 0
                neural_network = FeedForwardNeuralNetwork(genome)
                self.neural_networks.append(neural_network)

            for neural_network in self.neural_networks:
                evaluation_function(neural_network)

            total_fitness = 0
            for species in self.species:
                species.step_generation()
                total_fitness += species.fitness

            self.genomes.sort(key=lambda genome: genome.fitness, reverse=True)
            champion = deepcopy(self.genomes[0])
            self.genomes.clear()

            for species in self.species:
                num_individuals = int(self.population_size * species.fitness / total_fitness)
                species.reproduce(num_individuals, self.genome_identifier)
                # self.genomes += species.children
                for genome in species.elites:
                    self.genomes.append(genome)
                for genome in species.children:
                    self.mutate_genome(genome)
                    self.genomes.append(genome)

            # update loop conditions
            self.generations_run += 1
            max_fitness = champion.fitness

            if champion.fitness > self.champion.fitness:
                self.champion = champion

            print("Best genome in generation {}: genome {}, fitness: {}".format(self.generations_run, self.champion.identifier, self.champion.fitness),file=output_stream)

        return self.champion

    # assume empty species but full genome list
    def set_species(self):

        for genome in self.genomes:

            compatible_species = ([species for species in self.species if species.is_compatible_with(genome)] + [None])[0]
            if compatible_species is None:
                self.species.append( Species(self.next_species_identifier, 1, genome) )
                self.next_species_identifier += 1
            else:
                compatible_species.add_genome(genome)

    def set_innovation_number(self, edge):

        matching_edge = ([matching_edge for matching_edge in self.innovations if edge.input_node_identifier == matching_edge.input_node_identifier and edge.output_node_identifier == matching_edge.output_node_identifier] + [None])[0]
        if matching_edge == None:

            edge.innovation_number = self.next_innovation_number

            self.innovations.append(edge)
            self.next_innovation_number += 1

        else:
            edge.innovation_number = matching_edge.innovation_number

    def __str__(self):

        return "Population of {} individuals. {} inputs, {} outputs".format(len(self.genomes), self.num_inputs, self.num_outputs)